"""
Shared Browser Manager (DrissionPage)

Manages a single ChromiumPage browser instance.
Supports multi-tab concurrency via DrissionPage's built-in tab management.
"""

import threading
import tempfile
import shutil
import os
import time
import signal
import subprocess
from typing import Optional, Any
from loguru import logger
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.errors import PageDisconnectedError

class SharedBrowserManager:
    """
    Manages a shared DrissionPage Chromium browser.
    """
    
    _instance: Optional["SharedBrowserManager"] = None
    _lock = threading.Lock()
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._page: Optional[ChromiumPage] = None
        self._starting = False
        self._tab_lock = threading.Lock()
        self._profile_dirs: list[str] = []

    @staticmethod
    def _env_float(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(str(os.environ.get(name, str(default))).strip())
        except Exception:
            value = default
        return max(low, min(high, value))

    @staticmethod
    def _list_hyw_browser_pids(profile_hint: str | None = None) -> list[int]:
        """Find browser processes launched by this project profile marker."""
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,command="],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []

        pids: list[int] = []
        for raw in (result.stdout or "").splitlines():
            row = str(raw or "").strip()
            if not row:
                continue
            parts = row.split(None, 1)
            if len(parts) != 2:
                continue
            pid_s, cmd = parts
            if "hyw-dp-" not in cmd:
                continue
            if "--user-data-dir=" not in cmd:
                continue
            if profile_hint and profile_hint not in cmd:
                continue
            try:
                pids.append(int(pid_s))
            except Exception:
                continue
        return sorted(set(pids))

    @staticmethod
    def _terminate_pids(pids: list[int], *, grace_seconds: float = 1.0) -> None:
        if not pids:
            return
        alive: list[int] = []
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                alive.append(pid)
            except Exception:
                continue
        if not alive:
            return

        deadline = time.time() + max(0.2, grace_seconds)
        still = alive[:]
        while still and time.time() < deadline:
            next_round: list[int] = []
            for pid in still:
                try:
                    os.kill(pid, 0)
                    next_round.append(pid)
                except Exception:
                    continue
            still = next_round
            if still:
                time.sleep(0.05)

        for pid in still:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                continue

    def _cleanup_stale_browser_processes(self, profile_hint: str | None = None) -> int:
        pids = self._list_hyw_browser_pids(profile_hint=profile_hint)
        if not pids:
            return 0
        self._terminate_pids(pids, grace_seconds=1.2)
        return len(pids)

    def _cleanup_profile_dir(self, profile_dir: str) -> None:
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            self._profile_dirs.remove(profile_dir)
        except Exception:
            pass

    def _spawn_chromium_with_timeout(self, co: ChromiumOptions, timeout_seconds: float) -> ChromiumPage:
        out: dict[str, Any] = {}
        err: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                out["page"] = ChromiumPage(addr_or_opts=co)
            except BaseException as exc:  # noqa: BLE001
                err["exc"] = exc

        t = threading.Thread(target=_runner, name="hyw-browser-launch", daemon=True)
        t.start()
        t.join(timeout=max(0.5, timeout_seconds))

        if t.is_alive():
            raise TimeoutError(f"ChromiumPage init timeout after {timeout_seconds:.1f}s")
        if "exc" in err:
            raise err["exc"]
        page = out.get("page")
        if page is None:
            raise RuntimeError("ChromiumPage init returned no page")
        return page

    @classmethod
    def get_instance(cls, headless: bool = True) -> "SharedBrowserManager":
        """Get or create the singleton SharedBrowserManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(headless=headless)
            return cls._instance
    
    def start(self) -> bool:
        """Start the Chromium browser."""
        if self._page is not None:
            # Check if alive
            try:
                if self._page.run_cdp('Browser.getVersion'):
                    return True
            except (PageDisconnectedError, Exception):
                self._page = None

        # If another thread is starting the browser, wait for it instead of failing fast.
        wait_timeout = self._env_float("HYW_BROWSER_START_WAIT_TIMEOUT", 10.0, 1.0, 60.0)
        wait_deadline = time.time() + wait_timeout
        while True:
            with self._lock:
                if not self._starting:
                    self._starting = True
                    break
                if self._page is not None:
                    return True
            if time.time() >= wait_deadline:
                logger.warning(
                    "SharedBrowserManager: wait for browser start lock timed out ({}s), recovering.",
                    round(wait_timeout, 2),
                )
                with self._lock:
                    self._starting = False
                wait_deadline = time.time() + wait_timeout
            time.sleep(0.05)
        
        try:
            logger.info("SharedBrowserManager: Starting DrissionPage browser...")
            last_error: Optional[Exception] = None
            max_attempts = 1
            try:
                max_attempts = int(str(os.environ.get("HYW_BROWSER_START_ATTEMPTS", "1")).strip() or "1")
            except Exception:
                max_attempts = 1
            max_attempts = max(1, min(3, max_attempts))
            launch_timeout = self._env_float("HYW_BROWSER_LAUNCH_TIMEOUT", 8.0, 1.0, 30.0)
            for attempt in range(1, max_attempts + 1):
                profile_dir = tempfile.mkdtemp(prefix=f"hyw-dp-{attempt}-")
                self._profile_dirs.append(profile_dir)
                try:
                    co = self._build_options(profile_dir)
                    self._page = self._spawn_chromium_with_timeout(co, launch_timeout)

                    # Show Landing Page
                    try:
                        import os
                        landing_path = os.path.join(os.path.dirname(__file__), 'landing.html')
                        if os.path.exists(landing_path):
                            self._page.get(f"file://{landing_path}")
                    except Exception as e:
                        logger.warning(f"SharedBrowserManager: Failed to show landing page: {e}")

                    logger.success(f"SharedBrowserManager: Browser ready (port={self._page.address})")
                    return True
                except Exception as e:
                    last_error = e
                    self._page = None
                    try:
                        cleaned = self._cleanup_stale_browser_processes(profile_hint=profile_dir)
                        if cleaned:
                            logger.warning(
                                "SharedBrowserManager: cleaned {} stale browser process(es) for attempt {}.",
                                cleaned,
                                attempt,
                            )
                    except Exception:
                        pass
                    self._cleanup_profile_dir(profile_dir)
                    logger.warning(f"SharedBrowserManager: Start attempt {attempt}/{max_attempts} failed: {e}")
            assert last_error is not None
            raise last_error
            
        except Exception as e:
            logger.error(f"SharedBrowserManager: Failed to start browser: {e}")
            self._page = None
            raise
        finally:
            self._starting = False

    def _build_options(self, profile_dir: str) -> ChromiumOptions:
        """Build resilient Chromium options for startup."""
        co = ChromiumOptions()
        co.headless(self.headless)
        co.auto_port()  # Auto find available port

        # Isolate user data dir to avoid conflicts with existing Chrome instances.
        # DrissionPage 4.1.1.2 has compatibility issues with set_user_data_path() on some systems.
        co.set_argument(f'--user-data-dir={profile_dir}')

        # Anti-detection and startup stability flags.
        co.set_argument('--no-sandbox')
        co.set_argument('--disable-gpu')
        co.set_argument('--disable-dev-shm-usage')
        if self.headless:
            co.set_argument('--headless=new')

        # Essential for loading local files and avoiding CORS issues
        co.set_argument('--allow-file-access-from-files')
        co.set_argument('--disable-web-security')
        # Hide scrollbars globally
        co.set_argument('--hide-scrollbars')
        # 十万的原因是滚动条屏蔽(大概吧)
        co.set_argument('--window-size=1280,800')
        return co
    
    @property
    def page(self) -> Optional[ChromiumPage]:
        """Get the main ChromiumPage instance."""
        if self._page is None:
            self.start()
        return self._page

    def new_tab(self, url: str = None) -> Any:
        """
        Thread-safe tab creation.
        DrissionPage is thread-safe for tab creation, so we call it directly
        to allow atomic creation+navigation (Target.createTarget with url)
        without blocking other threads.
        """
        page = self.page
        if not page:
             raise RuntimeError("Browser not available")
        
        # Direct call allows Chrome to handle creation and navigation atomically and concurrently
        return page.new_tab(url)

    
    def close(self):
        """Shutdown the browser."""
        known_profiles: list[str] = []
        with self._lock:
            if self._page:
                try:
                    self._page.quit()
                    logger.info("SharedBrowserManager: Browser closed.")
                except BaseException as e:
                    logger.warning(f"SharedBrowserManager: Error closing browser: {e}")
                finally:
                    self._page = None
            if self._profile_dirs:
                known_profiles = list(self._profile_dirs)
                for path in self._profile_dirs:
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                    except Exception:
                        pass
                self._profile_dirs.clear()
        # Last-resort cleanup for orphan browser processes of this instance only.
        cleaned_total = 0
        for profile in known_profiles:
            try:
                cleaned_total += self._cleanup_stale_browser_processes(profile_hint=profile)
            except Exception:
                continue
        if cleaned_total:
            logger.info("SharedBrowserManager: cleaned {} orphan browser process(es) on close.", cleaned_total)

    @staticmethod
    def hide_scrollbars(page: ChromiumPage):
        """
        Robustly hide scrollbars using CDP commands AND CSS injection.
        This provides double protection against scrollbar gutters.
        """
        try:
            # 1. CDP Command
            page.run_cdp('Emulation.setScrollbarsHidden', hidden=True)
            
            # 2. CSS Injection (Standard + Webkit)
            css = """
                ::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
                * { -ms-overflow-style: none !important; scrollbar-width: none !important; }
            """
            # Inject into current page
            page.run_js(f"""
                const style = document.createElement('style');
                style.textContent = `{css}`;
                document.head.appendChild(style);
            """)
            
            logger.debug("SharedBrowserManager: Scrollbars hidden via CDP + CSS.")
        except Exception as e:
            logger.warning(f"SharedBrowserManager: Failed to hide scrollbars: {e}")


# Module-level singleton accessor
_shared_manager: Optional[SharedBrowserManager] = None


def get_shared_browser_manager(headless: bool = True) -> SharedBrowserManager:
    """
    Get or create the shared browser manager.
    """
    global _shared_manager
    
    if _shared_manager is None:
        _shared_manager = SharedBrowserManager.get_instance(headless=headless)
        _shared_manager.start()
    
    return _shared_manager


def close_shared_browser():
    """Close the shared browser manager."""
    global _shared_manager
    if _shared_manager:
        _shared_manager.close()
        _shared_manager = None

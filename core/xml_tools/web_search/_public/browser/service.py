"""
Browser Service (DrissionPage)

Provides page fetching and screenshot capabilities using DrissionPage.
"""

import asyncio
import base64
import threading
import time
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from loguru import logger
import trafilatura
from PIL import Image
from io import BytesIO

# Import intelligent completeness checker
from ..crawling.completeness import CompletenessChecker, trigger_lazy_load
from ..crawling.models import CrawlConfig

class ScreenshotService:
    """
    Browser Service using DrissionPage.
    """
    
    def __init__(self, headless: bool = True, auto_start: bool = False):
        self.headless = headless
        self._manager = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        if auto_start:
            self._ensure_ready()

    def _navigate_tab(self, tab: Any, url: str, timeout: float = 20.0):
        """Robust navigation: avoid about:blank by retrying with JS redirect."""
        target = str(url or "").strip()
        if not target:
            return

        nav_timeout = max(3.0, float(timeout))
        try:
            tab.get(target, show_errmsg=False, retry=0, interval=0.2, timeout=nav_timeout)
        except TypeError:
            tab.get(target)
        except Exception as e:
            logger.warning(f"ScreenshotService: tab.get failed once, retry via JS redirect: {e}")

        try:
            current = str(tab.url or "").strip().lower()
        except Exception:
            current = ""

        if current in {"", "about:blank", "chrome://new-tab-page/"}:
            try:
                tab.run_js(f"window.location.href = {target!r};")
                tab.wait.doc_loaded(timeout=min(nav_timeout, 8.0))
            except Exception as e:
                logger.warning(f"ScreenshotService: JS redirect failed: {e}")

    def _get_tab(self, url: str, timeout: float = 20.0) -> Any:
        """Create a new tab and navigate to URL."""
        self._ensure_ready()
        tab = self._manager.new_tab()
        self._navigate_tab(tab, url, timeout=timeout)
        return tab

    def _release_tab(self, tab: Any):
        """Close tab after use."""
        if not tab: return
        try:
            tab.close()
        except:
            pass

    def _collect_loaded_image_b64(self, tab: Any, max_images: int = 8, settle_timeout: float = 1.6) -> List[str]:
        """
        Collect image bytes already loaded by browser network stack.
        Requires tab.listen.start(res_type='Image') to be started before navigation.
        """
        picked: List[str] = []
        seen_url: set[str] = set()
        seen_sig: set[str] = set()

        try:
            tab.listen.wait_silent(timeout=max(0.2, float(settle_timeout)), targets_only=True, limit=0)
        except Exception:
            pass

        while len(picked) < max(1, int(max_images)):
            try:
                packet = tab.listen.wait(count=1, timeout=0.06, fit_count=False)
            except Exception:
                break
            if not packet:
                break

            rows = packet if isinstance(packet, list) else [packet]
            for row in rows:
                if row is False:
                    continue
                try:
                    if bool(getattr(row, "is_failed", False)):
                        continue
                    resource_type = str(getattr(row, "resourceType", "") or "").upper()
                    if resource_type and resource_type != "IMAGE":
                        continue

                    image_url = str(getattr(row, "url", "") or "").strip()
                    if not image_url or image_url in seen_url:
                        continue

                    response = getattr(row, "response", None)
                    if response is None:
                        continue

                    mime = str(getattr(response, "mimeType", "") or "").strip().lower()
                    if not mime.startswith("image/"):
                        continue

                    body = response.body
                    if not isinstance(body, (bytes, bytearray)):
                        continue
                    raw = bytes(body)
                    if len(raw) < 8 * 1024:
                        continue

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            img = Image.open(BytesIO(raw))
                        if img.mode == "P" and "transparency" in getattr(img, "info", {}):
                            img = img.convert("RGBA")
                        if img.mode in {"RGBA", "LA", "P"}:
                            img = img.convert("RGB")
                        if int(getattr(img, "width", 0) or 0) < 120 or int(getattr(img, "height", 0) or 0) < 80:
                            continue
                        if img.width > 1600:
                            ratio = 1600.0 / float(max(1, img.width))
                            img = img.resize((1600, max(1, int(float(img.height) * ratio))))
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=85, optimize=True)
                        image_b64 = base64.b64encode(buf.getvalue()).decode()
                    except Exception:
                        image_b64 = base64.b64encode(raw).decode()

                    sig = image_b64[:120]
                    if sig in seen_sig:
                        continue
                    seen_sig.add(sig)
                    seen_url.add(image_url)
                    picked.append(image_b64)
                    if len(picked) >= max_images:
                        break
                except Exception:
                    continue

        return picked

    async def search_via_page_input_batch(self, queries: List[str], url: str, selector: str = "#input") -> List[Dict[str, Any]]:
        """
        Execute concurrent searches using page inputs.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._search_via_page_input_batch_sync,
            queries, url, selector
        )

    def _search_via_page_input_batch_sync(self, queries: List[str], url: str, selector: str) -> List[Dict[str, Any]]:
        """Sync batch execution - create tabs sequentially, search in parallel."""
        results = [None] * len(queries)
        tabs = []
        
        # Phase 1: Get/create tabs SEQUENTIALLY (DrissionPage isn't thread-safe for new_tab)
        target_url = url or "https://www.google.com"
        logger.info(f"ScreenshotService: Acquiring {len(queries)} tabs for parallel search...")
        
        for i in range(len(queries)):
            tab = None
            # Try to get from pool first (using shared logic now)
            try:
                tab = self._get_tab(target_url, timeout=10.0)
            except Exception as e:
                logger.warning(f"ScreenshotService: Batch search tab creation failed: {e}")
            

            
            tabs.append(tab)
        
        logger.info(f"ScreenshotService: {len(tabs)} tabs ready, starting parallel searches...")
        
        # Phase 2: Execute searches in PARALLEL
        def run_search(index, tab, query):
            try:
                logger.debug(f"Search[{index}]: Starting for '{query}' on {tab.url}")
                
                # Wait for page to be ready first
                try:
                    tab.wait.doc_loaded(timeout=10)
                except:
                    pass
                
                # Find input element with wait
                logger.debug(f"Search[{index}]: Looking for input with selector '{selector}'")
                ele = tab.ele(selector, timeout=5)
                if not ele:
                    logger.debug(f"Search[{index}]: Primary selector failed, trying fallbacks")
                    for fallback in ["textarea[name='q']", "#APjFqb", "input[name='q']", "input[type='text']"]:
                        ele = tab.ele(fallback, timeout=2)
                        if ele:
                            logger.debug(f"Search[{index}]: Found input with fallback '{fallback}'")
                            break
                
                if not ele:
                    logger.error(f"Search[{index}]: No input element found on {tab.url}!")
                    results[index] = {"content": "Error: input not found", "title": "Error", "url": tab.url, "html": tab.html[:5000]}
                    return

                logger.debug(f"Search[{index}]: Typing query...")
                ele.input(query)
                
                logger.debug(f"Search[{index}]: Pressing Enter...")
                tab.actions.key_down('enter').key_up('enter')
                
                logger.debug(f"Search[{index}]: Waiting for search results...")
                tab.wait.doc_loaded(timeout=10)
                # Reduced settle wait for extraction
                time.sleep(0.1)
                
                logger.debug(f"Search[{index}]: Extracting content...")
                html = tab.html
                content = trafilatura.extract(
                    html, include_links=True, include_images=True, include_comments=False,
                    include_tables=True, favor_precision=False, output_format="markdown"
                ) or ""
                
                logger.info(f"ScreenshotService: Search '{query}' completed -> {tab.url}")
                
                results[index] = {
                    "content": content,
                    "html": html,
                    "title": tab.title,
                    "url": tab.url,
                    "images": []
                }
                
            except Exception as e:
                logger.error(f"ScreenshotService: Search error for '{query}': {e}")
                results[index] = {"content": f"Error: {e}", "title": "Error", "url": "", "html": ""}
            finally:
                self._release_tab(tab)

        threads = []
        for i, (tab, query) in enumerate(zip(tabs, queries)):
            t = threading.Thread(target=run_search, args=(i, tab, query))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
            
        return results
    
    def _ensure_ready(self):
        """Ensure shared browser is ready."""
        from .manager import get_shared_browser_manager
        self._manager = get_shared_browser_manager(headless=self.headless)

    async def fetch_page(self, url: str, timeout: float = 10.0, include_screenshot: bool = True) -> Dict[str, Any]:
        """
        Fetch page content (and optionally screenshot).
        Runs in a thread executor to avoid blocking the async loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_page_sync,
            url,
            timeout,
            include_screenshot
        )

    async def search_via_address_bar(self, query: str, timeout: float = 20.0) -> Dict[str, Any]:
        """
        Search using browser's address bar (uses browser's default search engine).
        Simulates: Ctrl+L (focus address bar) -> type query -> Enter
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._search_via_address_bar_sync,
            query,
            timeout
        )
    
    def _search_via_address_bar_sync(self, query: str, timeout: float) -> Dict[str, Any]:
        """Synchronous address bar search logic."""
        if not query:
            return {"content": "Error: missing query", "title": "Error", "url": "", "html": ""}
        
        tab = None
        try:
            self._ensure_ready()
            page = self._manager.page
            if not page:
                return {"content": "Error: Browser not available", "title": "Error", "url": "", "html": ""}
            
            # Open new blank tab
            tab = page.new_tab()
            
            # Direct navigation to avoid focus stealing (Cmd+L/Ctrl+L forces window focus)
            import urllib.parse
            # Using DuckDuckGo as configured in entari.yml
            search_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
            tab.get(search_url)
            
            # Wait for page to load
            try:
                tab.wait.doc_loaded(timeout=timeout)
                # Reduced wait for initial results
                time.sleep(0.2)
            except:
                pass
            
            html = tab.html
            title = tab.title
            final_url = tab.url
            
            # Extract content
            content = trafilatura.extract(
                html, include_links=True, include_images=True, include_comments=False,
                include_tables=True, favor_precision=False, output_format="markdown"
            ) or ""
            
            logger.info(f"ScreenshotService: Address bar search completed -> {final_url}")
            
            return {
                "content": content,
                "html": html,
                "title": title,
                "url": final_url,
                "images": []
            }
            
        except Exception as e:
            logger.error(f"ScreenshotService: Address bar search failed: {e}")
            return {"content": f"Error: search failed ({e})", "title": "Error", "url": "", "html": ""}
        finally:
            if tab:
                try: tab.close()
                except: pass

    def _scroll_to_bottom(self, tab, step: int = 800, delay: float = 2.0, timeout: float = 10.0):
        """
        Scroll down gradually to trigger lazy loading.
        
        Args:
            delay: Max wait time per scroll step (seconds) if images aren't loading.
        """
        import time
        start = time.time()
        current_pos = 0
        try:
            while time.time() - start < timeout:
                # Scroll down
                current_pos += step
                tab.run_js(f"window.scrollTo(0, {current_pos});")
                
                # Active Wait: Check if images in viewport are loaded
                # Poll every 100ms, up to 'delay' seconds
                wait_start = time.time()
                while time.time() - wait_start < delay:
                    all_loaded = tab.run_js("""
                        return (async () => {
                            const imgs = Array.from(document.querySelectorAll('img'));
                            const viewportHeight = window.innerHeight;
                            
                            // 1. Identify images currently in viewport
                            const visibleImgs = imgs.filter(img => {
                                const rect = img.getBoundingClientRect();
                                return (rect.top < viewportHeight && rect.bottom > 0) && (rect.width > 0 && rect.height > 0);
                            });
                            
                            if (visibleImgs.length === 0) return true;

                            // 2. Check loading status using decode() AND heuristic for placeholders
                            // Some sites load a tiny blurred placeholder first. 
                            const checks = visibleImgs.map(img => {
                                // Enhanced placeholder detection (matching completeness.py)
                                const dataSrc = img.getAttribute('data-src') || img.getAttribute('data-original') || 
                                                img.getAttribute('data-lazy-src') || img.getAttribute('data-lazy') || '';
                                const className = (typeof img.className === 'string' ? img.className : '').toLowerCase();
                                const loadingAttr = img.getAttribute('loading') || '';
                                const src = img.src || '';
                                
                                const isPlaceholder = (
                                    // data-src not yet loaded
                                    (dataSrc && img.src !== dataSrc) ||
                                    // Natural size much smaller than display (blurred placeholder)
                                    (img.naturalWidth < 50 && img.clientWidth > 100) ||
                                    (img.naturalWidth < 100 && img.clientWidth > 200 && img.naturalWidth * 4 < img.clientWidth) ||
                                    // Lazy-loading class indicators
                                    className.includes('lazy') ||
                                    className.includes('lazyload') ||
                                    className.includes('lozad') ||
                                    // CSS blur filter applied
                                    (window.getComputedStyle(img).filter || '').includes('blur') ||
                                    // loading="lazy" + not complete
                                    (loadingAttr === 'lazy' && !img.complete)
                                );
                                
                                if (isPlaceholder) {
                                    // If it looks like a placeholder, we return false (not loaded)
                                    // unless it stays like this for too long (handled by outer timeout)
                                    return Promise.resolve(false);
                                }

                                if (img.complete && img.naturalHeight > 0) return Promise.resolve(true); 
                                
                                return img.decode().then(() => true).catch(() => false); 
                            });
                            
                            // Race against a small timeout to avoid hanging on one broken image
                            const allDecoded = Promise.all(checks);
                            const timeout = new Promise(resolve => setTimeout(() => resolve(false), 500));
                            
                            // If any check returned false (meaning placeholder or not decoded), result is false
                            return Promise.race([allDecoded, timeout]).then(results => {
                                if (!Array.isArray(results)) return results === true;
                                return results.every(res => res === true);
                            }); 
                        })();
                    """)
                    if all_loaded:
                        break
                    time.sleep(0.1)

                # Check if reached bottom
                height = tab.run_js("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
                if current_pos >= height:
                    break
            
            # Ensure final layout settle
            time.sleep(0.2)
            
        except Exception as e:
            logger.warning(f"ScreenshotService: Scroll failed: {e}")

    def _extract_serp_markdown(self, tab: Any, page_url: str, max_results: int = 30) -> str:
        """
        Extract search-result links from rendered SERP DOM.
        Fallback for pages where trafilatura keeps too little useful content.
        """
        js_code = """
            (() => {
                const normalize = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                const root = document.querySelector('#react-layout')
                    || document.querySelector('[data-testid=\"mainline\"]')
                    || document.body;

                const decodeDdgUrl = (raw) => {
                    if (!raw) return '';
                    let absolute = '';
                    try {
                        absolute = raw.startsWith('http') ? raw : new URL(raw, location.origin).toString();
                    } catch (_) {
                        return '';
                    }
                    try {
                        const parsed = new URL(absolute);
                        const uddg = parsed.searchParams.get('uddg');
                        return uddg || absolute;
                    } catch (_) {
                        return absolute;
                    }
                };

                const noiseFragments = [
                    '仅包含此网站的结果',
                    '重新搜索，不包含此网站',
                    '在所有结果中屏蔽此网站',
                    '分享有关该网站的反馈',
                ];

                const rows = [];
                const seen = new Set();
                const cards = Array.from(
                    root.querySelectorAll('article[data-testid=\"result\"], [data-testid=\"result\"]')
                );

                for (const card of cards) {
                    const titleA = card.querySelector('a[data-testid=\"result-title-a\"], h2 a, h3 a');
                    if (!titleA) continue;

                    const urlA = card.querySelector('a[data-testid=\"result-extras-url-link\"]') || titleA;
                    const rawHref = (titleA.getAttribute('href') || urlA.getAttribute('href') || '').trim();
                    const finalUrl = decodeDdgUrl(rawHref);
                    if (!/^https?:/i.test(finalUrl)) continue;

                    let host = '';
                    try {
                        host = new URL(finalUrl).hostname || '';
                    } catch (_) {}
                    if (!host || /duckduckgo\\.com$/i.test(host)) continue;

                    const title = normalize(titleA.textContent || '');
                    if (!title) continue;

                    const displayUrl = normalize(urlA.textContent || '');
                    let cardText = normalize(card.innerText || card.textContent || '');
                    for (const frag of noiseFragments) {
                        cardText = normalize(cardText.split(frag).join(' '));
                    }

                    let snippet = cardText;
                    const titlePos = snippet.indexOf(title);
                    if (titlePos >= 0) {
                        snippet = normalize(snippet.slice(titlePos + title.length));
                    }
                    if (displayUrl) {
                        snippet = normalize(snippet.split(displayUrl).join(' '));
                    }
                    if (snippet.length > 500) {
                        snippet = snippet.slice(0, 500).trim();
                    }

                    const dedupKey = `${finalUrl}::${title}`.toLowerCase();
                    if (seen.has(dedupKey)) continue;
                    seen.add(dedupKey);
                    rows.push({
                        title: title.slice(0, 220),
                        url: finalUrl,
                        host,
                        display_url: displayUrl,
                        snippet,
                    });
                }

                // Fallback for engines/pages without result cards.
                if (!rows.length) {
                    const anchors = Array.from(root.querySelectorAll('a[href]'));
                    for (const a of anchors) {
                        const title = normalize(a.textContent || '');
                        if (!title) continue;
                        const finalUrl = decodeDdgUrl((a.getAttribute('href') || '').trim());
                        if (!/^https?:/i.test(finalUrl)) continue;
                        let host = '';
                        try { host = new URL(finalUrl).hostname || ''; } catch (_) {}
                        if (!host || /duckduckgo\\.com$/i.test(host)) continue;
                        const key = `${finalUrl}::${title}`.toLowerCase();
                        if (seen.has(key)) continue;
                        seen.add(key);
                        rows.push({
                            title: title.slice(0, 220),
                            url: finalUrl,
                            host,
                            display_url: host,
                            snippet: '',
                        });
                    }
                }

                const related = Array.from(
                    document.querySelectorAll('.related-searches__item a, .related-searches a')
                )
                    .map((a) => normalize(a.textContent || ''))
                    .filter((x) => !!x);

                return { rows, related };
            })()
        """

        try:
            payload = tab.run_js(js_code, as_expr=True) or {}
        except Exception as e:
            logger.warning(f"ScreenshotService: SERP DOM extract failed: {e}")
            return ""

        if not isinstance(payload, dict):
            return ""

        rows = payload.get("rows")
        related = payload.get("related")
        if not isinstance(rows, list):
            return ""

        cleaned: List[Dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in rows:
            if not isinstance(item, dict):
                continue
            raw_url = str(item.get("url") or "").strip()
            if not raw_url:
                continue
            try:
                parsed = urllib.parse.urlparse(raw_url)
            except Exception:
                continue
            if parsed.scheme not in {"http", "https"}:
                continue
            if raw_url in seen_urls:
                continue
            seen_urls.add(raw_url)
            title = str(item.get("title") or "").strip() or (parsed.hostname or "No Title")
            host = str(item.get("host") or parsed.hostname or "").strip()
            display_url = str(item.get("display_url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            cleaned.append(
                {
                    "title": title,
                    "url": raw_url,
                    "host": host,
                    "display_url": display_url,
                    "snippet": snippet,
                }
            )
            if len(cleaned) >= max(1, int(max_results)):
                break

        if len(cleaned) < 5:
            return ""

        lines = ["# Search Results", "", f"Source: {page_url}", ""]
        for index, row in enumerate(cleaned, start=1):
            title = row["title"].replace("[", "\\[").replace("]", "\\]").strip()
            host = row["host"].strip()
            url = row["url"].strip()
            display_url = row.get("display_url", "").strip()
            snippet = row.get("snippet", "").strip()
            lines.append(f"{index}. [{title}]({url})")
            if display_url:
                lines.append(f"   {display_url}")
            elif host:
                lines.append(f"   {host}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        related_terms: List[str] = []
        if isinstance(related, list):
            seen_related: set[str] = set()
            for item in related:
                text = str(item or "").strip()
                if not text:
                    continue
                if text in seen_related:
                    continue
                seen_related.add(text)
                related_terms.append(text)
                if len(related_terms) >= 12:
                    break

        if related_terms:
            lines.append("## Related Searches")
            lines.append("")
            for term in related_terms:
                lines.append(f"- {term}")
        return "\n".join(lines).strip()

    def _fetch_page_sync(self, url: str, timeout: float, include_screenshot: bool) -> Dict[str, Any]:
        """Synchronous fetch logic."""
        if not url:
            return {"content": "Error: missing url", "title": "Error", "url": ""}
        
        tab = None
        image_listener_started = False
        try:
            self._ensure_ready()
            page = self._manager.page
            if not page:
                return {"content": "Error: Browser not available", "title": "Error", "url": url}
            
            tab = page.new_tab()
            try:
                tab.listen.clear()
                tab.listen.start(res_type="Image")
                image_listener_started = True
            except Exception:
                image_listener_started = False

            self._navigate_tab(tab, url, timeout=timeout)
            
            # Wait logic - optimized for search pages
            is_search_page = any(s in url.lower() for s in ['search', 'bing.com', 'duckduckgo', 'google.com/search', 'searx'])
            if is_search_page:
                # Optimized waiting: Rapidly poll for ACTUAL results > 0
                start_time = time.time()
                
                # Special fast-path for DDG Lite (HTML only, no JS rendering needed)
                if 'lite.duckduckgo' in url:
                    # just wait for body, it's static HTML
                     try:
                         tab.wait.doc_loaded(timeout=timeout)
                     except: pass
                     # Sleep tiny bit to ensure render
                     time.sleep(0.5)
                else:
                    while time.time() - start_time < timeout:
                        found_results = False
                        try:
                            if 'google' in url.lower():
                                # Check if we have any result items (.g, .MjjYud) or the main container (#search)
                                # Using checks with minimal timeout to allow fast looping
                                if tab.ele('.g', timeout=0.1) or tab.ele('.MjjYud', timeout=0.1) or tab.ele('#search', timeout=0.1):
                                    found_results = True
                            elif 'bing' in url.lower():
                                if tab.ele('.b_algo', timeout=0.1) or tab.ele('#b_results', timeout=0.1):
                                    found_results = True
                            elif 'duckduckgo' in url.lower():
                                if tab.ele('.result', timeout=0.1) or tab.ele('#react-layout', timeout=0.1):
                                    found_results = True
                            else:
                                # Generic fallback: wait for body to be populated
                                if tab.ele('body', timeout=0.1):
                                    found_results = True
                        except:
                            pass
                        
                        if found_results:
                            break
                        time.sleep(0.05)  # Faster polling (50ms) as requested
            else:
                # 1. Wait for document to settle (Fast Dynamic Wait)
                try:
                    tab.wait.doc_loaded(timeout=timeout)
                except: pass

            html = tab.html
            title = tab.title
            final_url = tab.url
            
            raw_screenshot_b64 = None
            if include_screenshot:
                try:
                    # Scrollbar Hiding Best Effort
                    from .manager import SharedBrowserManager
                    SharedBrowserManager.hide_scrollbars(tab)
                    
                    # Inject CSS
                    tab.run_js("""
                        const style = document.createElement('style');
                        style.textContent = `
                            ::-webkit-scrollbar { display: none !important; }
                            html, body { -ms-overflow-style: none !important; scrollbar-width: none !important; }
                        `;
                        document.head.appendChild(style);
                        document.documentElement.style.overflow = 'hidden';
                        document.body.style.overflow = 'hidden';
                    """)
                    
                    raw_screenshot_b64 = tab.get_screenshot(as_base64='jpg', full_page=False)
                except Exception as e:
                    logger.warning(f"ScreenshotService: Failed to capture screenshot: {e}")

            # Extract content
            content = trafilatura.extract(
                html, include_links=True, include_images=True, include_comments=False, 
                include_tables=True, favor_precision=False, output_format="markdown"
            ) or ""
            content = content.strip()

            # SERP pages often contain sparse/fragmented markdown under trafilatura;
            # supplement with a DOM-based link extraction when content is too short.
            if is_search_page and len(content) < 700:
                serp_md = self._extract_serp_markdown(tab=tab, page_url=str(final_url or url), max_results=30)
                if serp_md:
                    logger.info(
                        "ScreenshotService: Using SERP DOM fallback markdown (trafilatura_len={}, serp_len={}) for {}",
                        len(content),
                        len(serp_md),
                        final_url,
                    )
                    content = serp_md

            # 2. Extract images already loaded in the current page:
            # prefer network listener packets, then fallback to DOM canvas extraction.
            images_b64 = []
            try:
                if image_listener_started:
                    images_b64 = self._collect_loaded_image_b64(tab, max_images=8, settle_timeout=1.6)
                    if images_b64:
                        logger.info(f"ScreenshotService: Collected {len(images_b64)} loaded images via Network listener")

                if not images_b64:
                    js_code = """
                        (async () => {
                            const blocklist = ['logo', 'icon', 'avatar', 'ad', 'pixel', 'tracker', 'button', 'menu', 'nav'];
                            const candidates = Array.from(document.querySelectorAll('img'));
                            const validImages = [];
                            
                            // Helper: Get base64 from loaded image via Canvas
                            const getBase64 = (img) => {
                                try {
                                    const canvas = document.createElement('canvas');
                                    canvas.width = img.naturalWidth;
                                    canvas.height = img.naturalHeight;
                                    const ctx = canvas.getContext('2d');
                                    ctx.drawImage(img, 0, 0);
                                    return canvas.toDataURL('image/jpeg').split(',')[1];
                                } catch(e) { return null; }
                            };

                            for (const img of candidates) {
                                if (validImages.length >= 8) break;
                                
                                if (img.naturalWidth < 100 || img.naturalHeight < 80) continue;
                                
                                const alt = (img.alt || '').toLowerCase();
                                const cls = (typeof img.className === 'string' ? img.className : '').toLowerCase();
                                const src = (img.src || '').toLowerCase();
                                
                                if (blocklist.some(b => alt.includes(b) || cls.includes(b) || src.includes(b))) continue;
                                
                                // 1. Try Canvas (Instant for loaded images)
                                if (img.complete && img.naturalHeight > 0) {
                                    const b64 = getBase64(img);
                                    if (b64) {
                                        validImages.push(b64);
                                        continue;
                                    }
                                }
                            }
                            return validImages;
                        })()
                    """
                    images_b64 = tab.run_js(js_code, as_expr=True) or []
                    if images_b64:
                        logger.info(f"ScreenshotService: Extracted {len(images_b64)} images for {url}")

            except Exception as e:
                logger.warning(f"ScreenshotService: Image extraction failed: {e}")

            return {
                "content": content,
                "html": html,
                "title": title,
                "url": final_url,
                "raw_screenshot_b64": raw_screenshot_b64,
                "images": images_b64
            }

        except Exception as e:
            logger.error(f"ScreenshotService: Failed to fetch {url}: {e}")
            return {"content": f"Error: fetch failed ({e})", "title": "Error", "url": url}
        finally:
            if tab:
               try:
                   tab.listen.stop()
               except Exception:
                   pass
               self._release_tab(tab)

    async def fetch_pages_batch(self, urls: List[str], timeout: float = 20.0, include_screenshot: bool = True) -> List[Dict[str, Any]]:
        """Fetch multiple pages concurrently."""
        if not urls: return []
        logger.info(f"ScreenshotService: Batch fetching {len(urls)} URLs (screenshots={include_screenshot})")
        tasks = [self.fetch_page(url, timeout, include_screenshot) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def screenshot_urls_batch(self, urls: List[str], timeout: float = 15.0, full_page: bool = True) -> List[Optional[str]]:
        """Take screenshots of multiple URLs concurrently."""
        if not urls: return []
        logger.info(f"ScreenshotService: Batch screenshot {len(urls)} URLs")
        tasks = [self.screenshot_url(url, timeout=timeout, full_page=full_page) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def screenshot_url(self, url: str, wait_load: bool = True, timeout: float = 15.0, full_page: bool = False, quality: int = 90, scale: int = 1) -> Optional[str]:
        """Screenshot URL (Async wrapper for sync). Returns base64 string only."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._screenshot_sync,
            url, wait_load, timeout, full_page, quality, scale, False  # extract_content=False
        )
        # Backward compatible: return just the screenshot for old callers
        if isinstance(result, dict):
            return result.get("screenshot_b64")
        return result

    async def screenshot_with_content(self, url: str, timeout: float = 15.0, max_content_length: int = 8000) -> Dict[str, Any]:
        """
        Screenshot URL and extract page content.
        
        Returns:
            Dict with:
                - screenshot_b64: base64 encoded screenshot
                - content: trafilatura extracted text (truncated to max_content_length)
                - title: page title
                - url: final URL
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._screenshot_sync,
            url, True, timeout, False, 80, 2, True  # quality=80 for balance, scale=2, extract_content=True
        )
        
        if not isinstance(result, dict):
            return {"screenshot_b64": result, "content": "", "title": "", "url": url}
        
        # Truncate content if needed
        content = result.get("content", "") or ""
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[内容已截断...]"
        result["content"] = content
        
        return result


    def _screenshot_sync(self, url: str, wait_load: bool, timeout: float, full_page: bool, quality: int, scale: int = 1, extract_content: bool = False) -> Any:
        """Synchronous screenshot. If extract_content=True, returns Dict else str."""
        if not url: 
            return {"screenshot_b64": None, "content": "", "title": "", "url": url} if extract_content else None
        tab = None
        capture_width = 3000  # Increased for more comfortable page size while maintaining high resolution
        
        try:
            self._ensure_ready()
            page = self._manager.page
            if not page: return None
            
            # Create blank tab first
            tab = page.new_tab()
            
            # Set viewport BEFORE navigation so page renders at target width from the start
            # This eliminates the need for post-load resize and reflow
            try:
                tab.run_cdp('Emulation.setDeviceMetricsOverride', 
                            width=capture_width, height=900, deviceScaleFactor=scale, mobile=False)
            except:
                pass
            
            # Now navigate to the URL - page will render at target width
            tab.get(url)
            
            # Network monitoring removed - not needed for simplified wait logic

            # Initialize crawl config for completeness checking (defined outside try for scope)
            crawl_config = CrawlConfig(
                scan_full_page=True,
                scroll_step=800,
                scroll_delay=0.5,
                scroll_timeout=20.0,  # Increased for lazy-loading pages
                image_load_timeout=3.0,  # Image loading timeout: 3 seconds
                image_stability_checks=3,
            )

            # === Start scrolling immediately to trigger lazy loading ===
            # Scroll first, then wait for DOM - this allows lazy loading to start in parallel
            logger.info(f"ScreenshotService: Starting lazy load scroll for {url} (before DOM wait)")
            trigger_lazy_load(tab, crawl_config)

            # Now wait for DOM to be ready (after scroll has triggered lazy loading)
            try:
                # Wait for full page load (including JS execution)
                try:
                    tab.wait.doc_loaded(timeout=timeout)
                except:
                    pass
                
                # Wait for actual content to appear (for CDN verification pages)
                # Simplified wait logic for screenshot: just check basic readiness
                
                for i in range(20):  # Max 20 iterations (~1s) - much faster
                    try:
                        state = tab.run_js('''
                            return {
                                ready: document.readyState === 'complete',
                                title: document.title,
                                text: document.body.innerText.substring(0, 500) || ""
                            };
                        ''') or {'ready': False, 'title': "", 'text': ""}
                        
                        is_ready = state.get('ready', False)
                        title = state.get('title', "").lower()
                        text_lower = state.get('text', "").lower()
                        text_len = len(text_lower)
                        
                        # Check for verification pages
                        is_verification = "checking your browser" in text_lower or \
                                        "just a moment" in text_lower or \
                                        "please wait" in text_lower or \
                                        "security check" in title or \
                                        "just a moment" in title or \
                                        "loading..." in title
                        
                        # Basic content check
                        has_content = text_len > 100
                        
                        # Pass if ready, not verification, and has content
                        if is_ready and not is_verification and has_content:
                            logger.debug(f"ScreenshotService: Page ready after {i * 0.05:.2f}s")
                            break
                        
                        time.sleep(0.05)
                    except Exception:
                        time.sleep(0.05)
                        continue
                
                # DEBUG: Save HTML to inspect what happened (in data dir)
                try:
                    import os
                    log_path = os.path.join(os.getcwd(), "data", "browser.log.html")
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"<!-- URL: {url} -->\n")
                        f.write(tab.html)
                except: pass

            except Exception as e:
                logger.warning(f"ScreenshotService: Page readiness check failed: {e}")
            
            # Scrollbar Hiding first (before any height calculation)
            from .manager import SharedBrowserManager
            SharedBrowserManager.hide_scrollbars(tab)
            
            # Scroll back to top
            tab.run_js("window.scrollTo(0, 0);")
            
            # Image loading monitoring with time tracking - DISABLED
            # No longer waiting for images to load
            # Initialize image tracking JavaScript
            # image_tracking_js = """
            # (() => {
            #     if (!window._imageLoadTracker) {
            #         window._imageLoadTracker = {
            #             startTime: Date.now(),
            #             images: new Map()
            #         };
            #         
            #         const imgs = Array.from(document.querySelectorAll('img'));
            #         const minSize = 50;
            #         
            #         imgs.forEach((img, idx) => {
            #             if (img.clientWidth < minSize && img.clientHeight < minSize) return;
            #             
            #             const src = img.src || img.getAttribute('data-src') || '';
            #             const key = `${idx}_${src.substring(0, 100)}`;
            #             
            #             if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
            #                 // Already loaded
            #                 window._imageLoadTracker.images.set(key, {
            #                     src: src.substring(0, 150),
            #                     status: 'loaded',
            #                     loadTime: 0,  // Already loaded before tracking
            #                     naturalSize: [img.naturalWidth, img.naturalHeight],
            #                     displaySize: [img.clientWidth, img.clientHeight]
            #                 });
            #             } else {
            #                 // Track loading
            #                 window._imageLoadTracker.images.set(key, {
            #                     src: src.substring(0, 150),
            #                     status: 'pending',
            #                     startTime: Date.now(),
            #                     naturalSize: [img.naturalWidth, img.naturalHeight],
            #                     displaySize: [img.clientWidth, img.clientHeight]
            #                 });
            #                 
            #                 // Add load event listener
            #                 img.addEventListener('load', () => {
            #                     const entry = window._imageLoadTracker.images.get(key);
            #                     if (entry && entry.status === 'pending') {
            #                         entry.status = 'loaded';
            #                         entry.loadTime = Date.now() - entry.startTime;
            #                     }
            #                 });
            #                 
            #                 img.addEventListener('error', () => {
            #                     const entry = window._imageLoadTracker.images.get(key);
            #                     if (entry && entry.status === 'pending') {
            #                         entry.status = 'failed';
            #                         entry.loadTime = Date.now() - entry.startTime;
            #                     }
            #                 });
            #             }
            #         });
            #     }
            #     
            #     // Return current status
            #     const results = [];
            #     window._imageLoadTracker.images.forEach((value, key) => {
            #         const entry = {
            #             src: value.src,
            #             status: value.status,
            #             loadTime: value.status === 'loaded' ? (value.loadTime || 0) : (Date.now() - value.startTime),
            #             naturalSize: value.naturalSize,
            #             displaySize: value.displaySize
            #         };
            #         results.push(entry);
            #     });
            #     
            #     return {
            #         total: results.length,
            #         loaded: results.filter(r => r.status === 'loaded').length,
            #         pending: results.filter(r => r.status === 'pending').length,
            #         failed: results.filter(r => r.status === 'failed').length,
            #         details: results
            #     };
            # })()
            # """
            # 
            # # Initialize tracking
            # tab.run_js(image_tracking_js)
            # 
            # # Monitor image loading with dynamic stop logic
            # check_interval = 0.2  # Check every 200ms
            # image_timeout = 3.0  # Image loading timeout: 3 seconds
            # monitoring_start = time.time()
            # loaded_times = []  # Track load times of completed images
            # 
            # logger.info(f"ScreenshotService: Starting image load monitoring (timeout={image_timeout}s)...")
            # 
            # while True:
            #     elapsed = time.time() - monitoring_start
            #     
            #     # Check timeout first
            #     if elapsed >= image_timeout:
            #         logger.info(f"ScreenshotService: Image loading timeout ({image_timeout}s) reached")
            #         break
            #     
            #     # Get current image status
            #     status = tab.run_js(image_tracking_js, as_expr=True) or {
            #         'total': 0, 'loaded': 0, 'pending': 0, 'failed': 0, 'details': []
            #     }
            #     
            #     # Log each image's status and load time
            #     for img_detail in status.get('details', []):
            #         src_short = img_detail.get('src', '')[:80]
            #         status_str = img_detail.get('status', 'unknown')
            #         load_time = img_detail.get('loadTime', 0)
            #         logger.info(
            #             f"ScreenshotService: Image [{status_str}] "
            #             f"loadTime={load_time:.0f}ms "
            #             f"src={src_short}"
            #         )
            #     
            #     # Collect load times of completed images
            #     loaded_times = [
            #         img.get('loadTime', 0) 
            #         for img in status.get('details', []) 
            #         if img.get('status') == 'loaded' and img.get('loadTime', 0) > 0
            #     ]
            #     
            #     pending_count = status.get('pending', 0)
            #     loaded_count = status.get('loaded', 0)
            #     
            #     # Check stop conditions
            #     if pending_count == 0:
            #         logger.info(f"ScreenshotService: All images loaded. Total: {status.get('total', 0)}, Loaded: {loaded_count}")
            #         break
            #     
            #     # Check dynamic stop condition (if we have loaded images to calculate average)
            #     if loaded_times:
            #         avg_load_time = sum(loaded_times) / len(loaded_times)
            #         max_wait_time = avg_load_time * 2
            #         
            #         # Check if any pending image has exceeded max wait time
            #         pending_images = [
            #             img for img in status.get('details', [])
            #             if img.get('status') == 'pending'
            #         ]
            #         
            #         should_stop = False
            #         for pending_img in pending_images:
            #             wait_time = pending_img.get('loadTime', 0)
            #             if wait_time >= max_wait_time:
            #                 should_stop = True
            #                 logger.info(
            #                     f"ScreenshotService: Stopping - pending image waited {wait_time:.0f}ms, "
            #                     f"exceeds 2x avg load time ({max_wait_time:.0f}ms, avg={avg_load_time:.0f}ms)"
            #                 )
            #                 break
            #         
            #         if should_stop:
            #             break
            #     
            #     # Wait before next check
            #     time.sleep(check_interval)
            
            # Now calculate final height ONCE after all content loaded
            # CompletenessChecker already verified height stability
            try:
                final_height = tab.run_js('''
                    return Math.max(
                        document.body.scrollHeight || 0,
                        document.documentElement.scrollHeight || 0,
                        document.body.offsetHeight || 0,
                        document.documentElement.offsetHeight || 0
                    );
                ''')
                h = min(int(final_height) + 50, 15000)
                tab.run_cdp('Emulation.setDeviceMetricsOverride', 
                            width=capture_width, height=h, deviceScaleFactor=1, mobile=False)
            except:
                pass
            
            # Final scroll to top
            tab.run_js("window.scrollTo(0, 0);")
            
            # Capture screenshot
            screenshot_b64 = tab.get_screenshot(as_base64='jpg', full_page=False)
            
            # Use Pillow for intelligent compression
            if screenshot_b64 and quality < 95: # Only compress if quality is not near maximum
                try:
                    img_bytes = base64.b64decode(screenshot_b64)
                    img = Image.open(BytesIO(img_bytes))
                    
                    # Convert to RGB if not already (some images might be RGBA, which JPEG doesn't support)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="WebP", quality=quality, optimize=True) # Output as WebP format
                    screenshot_b64 = base64.b64encode(output_buffer.getvalue()).decode()
                    logger.debug(f"ScreenshotService: Applied Pillow compression with quality={quality}")
                except Exception as e:
                    logger.warning(f"ScreenshotService: Pillow compression failed: {e}")
            
            # Extract content if requested
            if extract_content:
                try:
                    html = tab.html
                    title = tab.title
                    final_url = tab.url
                    
                    # Minimal trafilatura settings to reduce token consumption
                    content = trafilatura.extract(
                        html,
                        include_links=False,     # No links to reduce tokens
                        include_images=False,    # No image descriptions
                        include_comments=False,  # No comments
                        include_tables=False,    # No tables (can be verbose)
                        favor_precision=True,    # Favor precision over recall
                        output_format="txt"      # Plain text (no markdown formatting)
                    ) or ""
                    
                    return {
                        "screenshot_b64": screenshot_b64,
                        "content": content,
                        "title": title,
                        "url": final_url
                    }
                except Exception as e:
                    logger.warning(f"ScreenshotService: Content extraction failed: {e}")
                    return {"screenshot_b64": screenshot_b64, "content": "", "title": "", "url": url}
            
            return screenshot_b64
                
        except Exception as e:
            logger.error(f"ScreenshotService: Screenshot URL failed: {e}")
            return {"screenshot_b64": None, "content": "", "title": "", "url": url} if extract_content else None
        finally:
            if tab:
                try: tab.close()
                except: pass


    async def execute_script(self, script: str) -> Dict[str, Any]:
        """
        Execute JavaScript in the current active page context.
        This reuses the shared browser instance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._execute_script_sync,
            script
        )

    def _execute_script_sync(self, script: str) -> Dict[str, Any]:
        """Synchronous JS execution."""
        try:
            self._ensure_ready()
            page = self._manager.page
            if not page:
                return {"success": False, "error": "Browser not available"}

            # Get current active tab or first tab
            # Fix: ChromiumPage object has no attribute 'tabs'
            # We use the page object itself as it represents the active tab controller
            tab = page
            if not tab:
                return {"success": False, "error": "No active tab"}

            logger.info(f"ScreenshotService: Executing JS on {tab.url}")

            # Execute JS
            result = tab.run_js(script)

            return {
                "success": True,
                "result": result,
                "url": tab.url,
                "title": tab.title
            }
        except Exception as e:
            logger.error(f"ScreenshotService: JS execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        self._executor.shutdown(wait=False)
        logger.info("ScreenshotService: Closed.")

    async def close_async(self):
        await self.close()

# Singleton
_screenshot_service: Optional[ScreenshotService] = None

def get_screenshot_service(headless: bool = True) -> ScreenshotService:
    global _screenshot_service
    if _screenshot_service is None:
        _screenshot_service = ScreenshotService(headless=headless, auto_start=False)
    return _screenshot_service

async def close_screenshot_service():
    global _screenshot_service
    if _screenshot_service:
        await _screenshot_service.close()
        _screenshot_service = None

def prestart_browser(headless: bool = True):
    svc = get_screenshot_service(headless=headless)
    try:
        svc._ensure_ready()
    except Exception:
        # Warmup is best-effort only.
        pass

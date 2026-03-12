"""
Page Completeness Checker

Multi-signal page readiness detection with focus on image loading guarantees.
Implements Crawl4AI's wait_for_images and networkidle concepts.
"""

import time
from typing import Any, Optional, Dict
from dataclasses import dataclass
from loguru import logger

from .models import CrawlConfig, CompletenessResult


# CSP-compliant JavaScript for image loading verification
# Based on Crawl4AI's wait_for_images logic
IMAGE_CHECK_JS = """
(() => {
    const results = {
        total: 0,
        loaded: 0,
        failed: 0,
        placeholders: 0,
        details: []
    };
    
    const imgs = Array.from(document.querySelectorAll('img'));
    const viewportHeight = window.innerHeight;
    const minSize = %d;  // Injected from Python
    
    for (const img of imgs) {
        // Skip tiny images (icons, tracking pixels)
        if (img.clientWidth < minSize && img.clientHeight < minSize) {
            continue;
        }
        
        // Skip images outside viewport (unless scan_full_page)
        const rect = img.getBoundingClientRect();
        const inViewport = rect.top < viewportHeight * 2 && rect.bottom > -viewportHeight;
        
        if (!inViewport && !%s) {  // scan_full_page injected
            continue;
        }
        
        results.total++;
        
        const src = img.src || img.getAttribute('data-src') || '';
        const dataSrc = img.getAttribute('data-src') || img.getAttribute('data-original') || 
                        img.getAttribute('data-lazy-src') || img.getAttribute('data-lazy') || '';
        const className = (typeof img.className === 'string' ? img.className : '').toLowerCase();
        const loadingAttr = img.getAttribute('loading') || '';
        
        // Enhanced placeholder detection for blurred preview images (like mcmod.cn)
        const isPlaceholder = (
            // 1. data-src exists but not yet loaded into src
            (dataSrc && img.src !== dataSrc) ||
            // 2. Natural size much smaller than display size (blurred placeholder)
            (img.naturalWidth < 50 && img.clientWidth > 100) ||
            (img.naturalWidth < 100 && img.clientWidth > 200 && img.naturalWidth * 4 < img.clientWidth) ||
            // 3. Common placeholder keywords in src
            src.includes('placeholder') ||
            src.includes('loading') ||
            src.includes('blank') ||
            // 4. SVG placeholder or 1x1 tracking pixel
            src.startsWith('data:image/svg+xml') ||
            (img.naturalWidth === 1 && img.naturalHeight === 1) ||
            // 5. Lazy-loading class indicators (common patterns)
            className.includes('lazy') ||
            className.includes('lazyload') ||
            className.includes('lozad') ||
            className.includes('b-lazy') ||
            // 6. Blur indicators (common for LQIP - Low Quality Image Placeholder)
            className.includes('blur') ||
            src.includes('blur') ||
            src.includes('thumb') ||
            src.includes('thumbnail') ||
            // 7. loading="lazy" + not complete (browser native lazy loading)
            (loadingAttr === 'lazy' && !img.complete) ||
            // 8. CSS blur filter applied (visual blurring)
            (window.getComputedStyle(img).filter || '').includes('blur')
        );
        
        if (isPlaceholder) {
            results.placeholders++;
            results.details.push({
                src: src.substring(0, 100),
                status: 'placeholder',
                natural: [img.naturalWidth, img.naturalHeight],
                display: [img.clientWidth, img.clientHeight]
            });
            continue;
        }
        
        // Check if fully loaded
        if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
            results.loaded++;
            results.details.push({
                src: src.substring(0, 100),
                status: 'loaded',
                natural: [img.naturalWidth, img.naturalHeight]
            });
        } else {
            results.failed++;
            results.details.push({
                src: src.substring(0, 100),
                status: 'pending',
                complete: img.complete,
                natural: [img.naturalWidth, img.naturalHeight]
            });
        }
    }
    
    return results;
})()
"""

# JavaScript for height stability check
HEIGHT_CHECK_JS = """
(() => {
    return {
        height: Math.max(
            document.body.scrollHeight || 0,
            document.documentElement.scrollHeight || 0,
            document.body.offsetHeight || 0,
            document.documentElement.offsetHeight || 0
        ),
        ready: document.readyState === 'complete'
    };
})()
"""


class CompletenessChecker:
    """
    Multi-signal page completeness verification.
    
    Signals checked:
    1. Image Loading (naturalWidth > 0, not placeholder)
    2. Height Stability (no layout shifts)
    3. DOM Ready State
    
    Usage:
        checker = CompletenessChecker(CrawlConfig())
        result = checker.check(tab)
        
        # Or wait until complete
        result = checker.wait_for_complete(tab)
    """
    
    def __init__(self, config: Optional[CrawlConfig] = None):
        self._config = config or CrawlConfig()
    
    def check(self, tab: Any) -> CompletenessResult:
        """
        Perform a single completeness check.
        
        Args:
            tab: DrissionPage tab object
            
        Returns:
            CompletenessResult with all signals
        """
        start = time.time()
        
        # Check images
        img_result = self._check_images(tab)
        
        # Check height
        height_result = self._check_height(tab)
        
        # Determine overall completeness
        # Complete if: all real images loaded AND height is stable
        total_pending = img_result.get('failed', 0) + img_result.get('placeholders', 0)
        all_images_loaded = total_pending == 0 or img_result.get('total', 0) == 0
        
        return CompletenessResult(
            is_complete=all_images_loaded and height_result.get('ready', False),
            total_images=img_result.get('total', 0),
            loaded_images=img_result.get('loaded', 0),
            failed_images=img_result.get('failed', 0),
            placeholder_images=img_result.get('placeholders', 0),
            height=height_result.get('height', 0),
            height_stable=True,  # Single check can't determine stability
            network_idle=True,   # TODO: network monitoring
            check_duration=time.time() - start
        )
    
    def wait_for_complete(
        self, 
        tab: Any, 
        timeout: Optional[float] = None
    ) -> CompletenessResult:
        """
        Wait for page to be fully loaded with image guarantees.
        
        Uses multi-signal approach:
        1. Poll image loading status
        2. Track height stability
        3. Respect timeout
        
        Args:
            tab: DrissionPage tab object
            timeout: Max wait time (default from config)
            
        Returns:
            CompletenessResult (may not be complete if timeout)
        """
        timeout = timeout or self._config.image_load_timeout
        start = time.time()
        
        height_history = []
        stable_count = 0
        last_result = None
        
        logger.debug(f"CompletenessChecker: Starting wait (timeout={timeout}s)")
        
        while time.time() - start < timeout:
            result = self.check(tab)
            last_result = result
            
            # Track height stability
            height_history.append(result.height)
            if len(height_history) > self._config.height_stability_checks:
                height_history.pop(0)
            
            # Check if height is stable
            height_stable = False
            if len(height_history) >= self._config.height_stability_checks:
                max_h = max(height_history)
                min_h = min(height_history)
                height_stable = (max_h - min_h) <= self._config.height_stability_threshold
            
            # Log progress
            elapsed = time.time() - start
            logger.debug(
                f"CompletenessChecker: [{elapsed:.1f}s] "
                f"images={result.loaded_images}/{result.total_images} "
                f"pending={result.failed_images} "
                f"placeholders={result.placeholder_images} "
                f"height={result.height} stable={height_stable}"
            )
            
            # Check if all images loaded AND height stable
            all_loaded = (
                result.failed_images == 0 and 
                result.placeholder_images == 0
            )
            
            if all_loaded and height_stable:
                stable_count += 1
                if stable_count >= self._config.image_stability_checks:
                    logger.info(
                        f"CompletenessChecker: Page complete! "
                        f"{result.loaded_images} images loaded, "
                        f"height={result.height}, "
                        f"took {elapsed:.2f}s"
                    )
                    return CompletenessResult(
                        is_complete=True,
                        total_images=result.total_images,
                        loaded_images=result.loaded_images,
                        failed_images=0,
                        placeholder_images=0,
                        height=result.height,
                        height_stable=True,
                        network_idle=True,
                        check_duration=elapsed
                    )
            else:
                stable_count = 0
            
            # Wait before next check
            time.sleep(self._config.image_check_interval)
        
        # Timeout reached
        elapsed = time.time() - start
        logger.warning(
            f"CompletenessChecker: Timeout after {elapsed:.1f}s! "
            f"images={last_result.loaded_images}/{last_result.total_images} "
            f"pending={last_result.failed_images}"
        )
        
        # Return last result with is_complete=False
        if last_result:
            return CompletenessResult(
                is_complete=False,
                total_images=last_result.total_images,
                loaded_images=last_result.loaded_images,
                failed_images=last_result.failed_images,
                placeholder_images=last_result.placeholder_images,
                height=last_result.height,
                height_stable=len(set(height_history)) == 1,
                network_idle=True,
                check_duration=elapsed
            )
        
        return CompletenessResult(
            is_complete=False,
            total_images=0, loaded_images=0, failed_images=0, placeholder_images=0,
            height=0, height_stable=False, network_idle=False,
            check_duration=elapsed
        )
    
    def _check_images(self, tab: Any) -> Dict[str, Any]:
        """Run image check JavaScript and return results."""
        try:
            js = IMAGE_CHECK_JS % (
                self._config.min_image_size,
                'true' if self._config.scan_full_page else 'false'
            )
            result = tab.run_js(js, as_expr=True)
            return result or {'total': 0, 'loaded': 0, 'failed': 0, 'placeholders': 0}
        except Exception as e:
            logger.warning(f"CompletenessChecker: Image check failed: {e}")
            return {'total': 0, 'loaded': 0, 'failed': 0, 'placeholders': 0}
    
    def _check_height(self, tab: Any) -> Dict[str, Any]:
        """Run height check JavaScript and return results."""
        try:
            result = tab.run_js(HEIGHT_CHECK_JS, as_expr=True)
            return result or {'height': 0, 'ready': False}
        except Exception as e:
            logger.warning(f"CompletenessChecker: Height check failed: {e}")
            return {'height': 0, 'ready': False}


def trigger_lazy_load(tab: Any, config: Optional[CrawlConfig] = None) -> None:
    """
    Scroll through page to trigger lazy-loaded images.
    
    Implements Crawl4AI's scan_full_page behavior.
    Strategy: Fast scroll with minimal delay (0.2s) per step to trigger network requests,
    then wait at the bottom for all images to settle.
    
    Args:
        tab: DrissionPage tab object
        config: Crawl configuration
    """
    config = config or CrawlConfig()
    if not config.scan_full_page:
        return
    
    start = time.time()
    current_pos = 0
    
    logger.info(f"CompletenessChecker: Starting lazy load scroll (fast scroll + final wait)")
    
    try:
        max_scroll_steps = 100
        step_count = 0
        
        # 1. Fast Scroll Phase
        while step_count < max_scroll_steps:
            step_count += 1
            current_pos += config.scroll_step
            tab.run_js(f"window.scrollTo(0, {current_pos});")
            
            # Simple fixed delay per step (0.2s) as requested
            time.sleep(0.2)
            
            # Check if reached bottom
            height = tab.run_js("""
                Math.max(
                    document.body.scrollHeight || 0,
                    document.documentElement.scrollHeight || 0
                )
            """, as_expr=True) or 0
            
            if current_pos >= height:
                logger.debug(f"CompletenessChecker: Reached bottom at position {current_pos}")
                break

        # 2. Wait Phase at Bottom (Wait for images to settle - reduced timeout)
        logger.debug("CompletenessChecker: Reached bottom, waiting for images to settle (max 2s)...")
        wait_start = time.time()
        max_wait_at_bottom = 2.0  # Reduced from 8s to 2s - scroll usually triggers loading quickly
        
        # Quick check: just verify images are not placeholders (simplified check)
        check_all_images_js = """
        (() => {
            const imgs = Array.from(document.querySelectorAll('img'));
            if (imgs.length === 0) return true;
            
            // Quick check: count non-placeholder images that are loaded
            let loaded_count = 0;
            let total_count = 0;
            
            for (const img of imgs) {
                // Skip tiny images
                if (img.clientWidth < 50 && img.clientHeight < 50) continue;
                
                total_count++;
                const dataSrc = img.getAttribute('data-src') || img.getAttribute('data-original') || '';
                const src = img.src || '';
                
                // Check if placeholder
                const isPlaceholder = (
                    (dataSrc && img.src !== dataSrc) ||
                    (img.naturalWidth < 50 && img.clientWidth > 100) ||
                    src.includes('placeholder') || src.includes('loading')
                );
                
                // If not placeholder and loaded, count it
                if (!isPlaceholder && img.complete && img.naturalWidth > 0) {
                    loaded_count++;
                }
            }
            
            // If most images are loaded (80%+), consider it done
            return total_count === 0 || (loaded_count / total_count) >= 0.8;
        })()
        """

        # Quick check loop with shorter interval
        check_count = 0
        max_checks = 4  # 2s / 0.5s = 4 checks max
        while check_count < max_checks:
            try:
                all_loaded = tab.run_js(check_all_images_js, as_expr=True)
                if all_loaded:
                    elapsed_wait = time.time() - wait_start
                    logger.debug(f"CompletenessChecker: Images settled at bottom in {elapsed_wait:.1f}s")
                    break
            except:
                pass
            time.sleep(0.5)
            check_count += 1
        
        # Scroll back to top
        tab.run_js("window.scrollTo(0, 0);")
        
        elapsed = time.time() - start
        logger.info(f"CompletenessChecker: Lazy load scroll complete - {step_count} steps in {elapsed:.1f}s")
        
    except Exception as e:
        logger.warning(f"CompletenessChecker: Lazy load scroll failed: {e}")

"""
DuckDuckGo Search Service

Provides web search functionality using DuckDuckGo.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from ..browser import BrowserService
from .engine import DuckDuckGoEngine


class DuckDuckGoSearchService:
    """
    Search service using DuckDuckGo.
    Handles search queries via DuckDuckGo web search.
    Delegates page fetching/screenshots to BrowserService.
    """

    def __init__(self, headless: bool = True, fetch_timeout: float = 20.0):
        self._headless = headless
        self._fetch_timeout = fetch_timeout
        self._engine = DuckDuckGoEngine()
        self._browser = BrowserService(headless=headless)

        logger.info("DuckDuckGoSearchService initialized")

    def _build_search_url(self, query: str, kl: Optional[str] = None, time_range: Optional[str] = None) -> str:
        return self._engine.build_url(query, kl=kl, time_range=time_range)

    async def search_batch(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        """Execute multiple searches concurrently."""
        logger.info(f"DuckDuckGoSearchService: Batch searching {len(queries)} queries in parallel...")
        tasks = [self.search(q) for q in queries]
        return await asyncio.gather(*tasks)

    async def search(self, query: str, kl: Optional[str] = None, time_range: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Main search entry point.
        Returns parsed search results only.
        """
        if not query:
            return []

        url = self._build_search_url(query, kl=kl, time_range=time_range)
        results = []

        try:
            logger.info(f"Search: query='{query}', kl='{kl or ''}', time_range='{time_range or ''}' -> {url}")
            page_data = await self._browser.fetch_page(url, include_screenshot=False)
            content = page_data.get("html", "") or page_data.get("content", "")

            logger.debug(f"Search: Raw content length = {len(content)} chars")
            if len(content) < 500:
                logger.warning(f"Search: Content too short, may be empty/blocked. First 500 chars: {content[:500]}")

            if content and not content.startswith("Error"):
                parsed = self._engine.parse(content)

                logger.info(f"Search: Engine parsed {len(parsed)} results from {len(content)} chars")

                logger.info(f"Search parsed {len(parsed)} results for '{query}'")

                # Add raw search page as hidden item for debug
                results.append({
                    "title": f"[DEBUG] Raw Search: {query}",
                    "url": url,
                    "content": content[:50000],
                    "_type": "search_raw_page",
                    "_hidden": True,
                })

                results.extend(parsed)
            else:
                logger.warning(f"Search failed/empty for '{query}': {content[:100]}")

            return results

        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return [{
                "title": f"Error Search: {query}",
                "url": url,
                "content": f"Error: {e}",
                "type": "search_raw_page",
                "_hidden": True
            }]

    # Delegate browser operations to BrowserService
    async def fetch_pages_batch(self, urls: List[str], include_screenshot: bool = True) -> List[Dict[str, Any]]:
        """Fetch multiple pages concurrently."""
        return await self._browser.fetch_pages_batch(urls, include_screenshot=include_screenshot)

    async def fetch_page(self, url: str, timeout: Optional[float] = None, include_screenshot: bool = True) -> Dict[str, Any]:
        """Fetch a single page for reading/extracting content."""
        return await self._browser.fetch_page(url, timeout=timeout, include_screenshot=include_screenshot)

    async def fetch_page_raw(self, url: str, timeout: Optional[float] = None, include_screenshot: bool = True) -> Dict[str, Any]:
        """Internal: Get raw data from browser service."""
        return await self._browser.fetch_page_raw(url, timeout=timeout, include_screenshot=include_screenshot)

    async def screenshot_url(self, url: str, full_page: bool = True) -> Optional[str]:
        """Capture a screenshot of a URL."""
        return await self._browser.screenshot_url(url, full_page=full_page)

    async def screenshot_with_content(self, url: str, max_content_length: int = 8000) -> Dict[str, Any]:
        """Capture screenshot and extract page content."""
        return await self._browser.screenshot_with_content(url, max_content_length=max_content_length)

    async def screenshot_urls_batch(self, urls: List[str], full_page: bool = True) -> List[Optional[str]]:
        """Capture screenshots of multiple URLs concurrently."""
        return await self._browser.screenshot_urls_batch(urls, full_page=full_page)

    async def execute_script(self, script: str) -> Dict[str, Any]:
        """Execute JavaScript in the current page context."""
        return await self._browser.execute_script(script)


# Backward compatibility alias
SearchService = DuckDuckGoSearchService

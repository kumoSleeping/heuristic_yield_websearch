"""Browser automation and rendering helpers."""

from .manager import (
    SharedBrowserManager,
    get_shared_browser_manager,
    close_shared_browser,
)

from .service import (
    ScreenshotService,
    get_screenshot_service,
    close_screenshot_service,
    prestart_browser,
)

from .renderer import (
    ContentRenderer,
    get_content_renderer,
    set_global_renderer,
)

# Aliases for cleaner API
BrowserManager = SharedBrowserManager
PageService = ScreenshotService
BrowserService = ScreenshotService
RenderService = ContentRenderer

__all__ = [
    # Browser Management
    "BrowserManager",
    "SharedBrowserManager",
    "get_shared_browser_manager",
    "close_shared_browser",

    # Page Service
    "PageService",
    "BrowserService",
    "ScreenshotService",
    "get_screenshot_service",
    "close_screenshot_service",
    "prestart_browser",
    
    # Render Service
    "RenderService",
    "ContentRenderer",
    "get_content_renderer",
    "set_global_renderer",
]

"""
Crawling Data Models

Core data structures for the intelligent crawling system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class CrawlConfig:
    """
    Crawling configuration with Crawl4AI-style parameters.
    
    Focuses on page completeness and image loading guarantees.
    """
    # Wait Strategy (Priority: Image Loading)
    wait_for_images: bool = True
    wait_until: str = "networkidle"  # domcontentloaded | networkidle | load
    delay_before_return: float = 0.1
    page_timeout: float = 30.0
    
    # Image Loading Specific
    image_load_timeout: float = 10.0  # Max wait for images
    image_stability_checks: int = 3   # Consecutive stable checks needed
    image_check_interval: float = 0.2 # Interval between checks
    min_image_size: int = 50          # Ignore images smaller than this
    
    # Scroll for lazy loading
    scan_full_page: bool = True
    scroll_step: int = 800
    scroll_delay: float = 0.5
    scroll_timeout: float = 15.0
    
    # Height Stability
    height_stability_checks: int = 3
    height_stability_threshold: int = 10  # pixels
    
    # Future: Adaptive Stop Logic
    confidence_threshold: float = 0.75
    min_gain_threshold: float = 0.1
    max_pages: int = 20


@dataclass
class CompletenessResult:
    """Result from completeness check."""
    is_complete: bool
    total_images: int
    loaded_images: int
    failed_images: int
    placeholder_images: int
    height: int
    height_stable: bool
    network_idle: bool
    check_duration: float
    
    @property
    def image_load_ratio(self) -> float:
        if self.total_images == 0:
            return 1.0
        return self.loaded_images / self.total_images


@dataclass
class PageResult:
    """Result from fetching a single page."""
    url: str
    final_url: str
    title: str
    html: str
    content: str  # Extracted markdown
    images: List[str] = field(default_factory=list)  # base64 images
    screenshot: Optional[str] = None
    
    # Quality Signals
    load_time: float = 0.0
    completeness: Optional[CompletenessResult] = None
    
    # Error handling
    error: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        if self.completeness is None:
            return False
        return self.completeness.is_complete

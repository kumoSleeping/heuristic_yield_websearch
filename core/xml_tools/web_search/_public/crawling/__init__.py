"""
hyw_core.crawling - Intelligent Web Crawling Module

Provides Crawl4AI-inspired adaptive crawling with:
- Page completeness guarantees (image loading verification)
- Content quality scoring
- Adaptive stop logic
"""

from .models import CrawlConfig, PageResult, CompletenessResult
from .completeness import CompletenessChecker

__all__ = [
    "CrawlConfig",
    "PageResult", 
    "CompletenessResult",
    "CompletenessChecker",
]

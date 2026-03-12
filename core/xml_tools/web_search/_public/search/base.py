"""
Search Engine Base Class

Abstract base for all search engine implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SearchEngine(ABC):
    @abstractmethod
    def build_url(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        """Build the search URL for the given query."""
        pass

    @abstractmethod
    def parse(self, content: str) -> List[Dict[str, Any]]:
        """Parse the raw HTML/Markdown content into a list of results."""
        pass

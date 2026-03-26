"""Thin package entrypoint that re-exports the core entari plugin implementation."""

from core import entari_plugin_hyw as _impl
from core.entari_plugin_hyw import *  # noqa: F401,F403

__version__ = _impl.__version__
__plugin__ = _impl.__plugin__

"""Importing this package registers every tool in tools.base.REGISTRY."""

from . import hr, knowledge  # noqa: F401
from .base import REGISTRY, ToolSpec, execute_tool, schemas_for  # noqa: F401

"""Local pathology AI service for PDC deployments."""

from .config import ServiceSettings
from .core import PathologyAIService, build_service

__all__ = ["PathologyAIService", "ServiceSettings", "build_service"]

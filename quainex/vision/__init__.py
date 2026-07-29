"""Screen understanding: screenshots, documents and window detection.

Phase 8. Vision through the AI provider rather than a local OCR stack, plus
free local window enumeration for questions that do not need a model.
"""

from quainex.vision.screen import ScreenAnalyst, WindowInfo

__all__ = ["ScreenAnalyst", "WindowInfo"]

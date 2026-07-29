"""Third-party service connectors.

Phase 9+. The Telegram bridge gives Quainex a phone interface that works from
anywhere without exposing this machine to inbound connections.
"""

from quainex.integrations.telegram import TELEGRAM_BLOCKED_INTENTS, TelegramBridge

__all__ = ["TELEGRAM_BLOCKED_INTENTS", "TelegramBridge"]

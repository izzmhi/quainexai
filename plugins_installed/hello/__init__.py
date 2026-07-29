"""Example Quainex plugin.

A worked reference for writing your own. Three things to notice:

1. **The module exposes a single ``plugin`` object** with an async
   ``handle(ctx, request)``. That is the entire interface.

2. **Capabilities come from ``ctx``, not from imports.** This plugin never
   imports Quainex internals, never touches the filesystem directly, and never
   constructs an API client. Anything it can do arrives through the context,
   which grants only what ``plugin.toml`` declared.

3. **It asks for the least it needs.** ``memory`` and ``ai``, nothing more. If
   this plugin later tried ``ctx.notify(...)`` it would raise, because the
   manifest you approved did not include it — which is the point.

The permission list is a statement of intent that the runtime holds you to. It
is not a sandbox; see ``quainex/plugins/manifest.py`` for the honest limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.plugins import PluginResponse

if TYPE_CHECKING:
    from quainex.plugins import PluginContext, PluginRequest


class HelloPlugin:
    """Greets the user and forwards questions to the model."""

    async def handle(self, ctx: PluginContext, request: PluginRequest) -> PluginResponse:
        """Handle one request.

        Args:
            ctx: The capabilities this plugin was granted.
            request: The action to run and its arguments.

        Returns:
            The response.
        """
        if request.action == "greet":
            return await self._greet(ctx, request.arguments.get("name"))
        if request.action == "ask":
            return await self._ask(ctx, request.arguments.get("question", ""))

        # The registry validates actions against the manifest before calling,
        # so this is defence in depth rather than the primary check.
        return PluginResponse(message=f"Unknown action '{request.action}'.")

    async def _greet(self, ctx: PluginContext, name: str | None) -> PluginResponse:
        """Greet the user, remembering their name across calls.

        Args:
            ctx: Granted capabilities.
            name: Name to remember, when supplied.

        Returns:
            The greeting.
        """
        if name:
            await ctx.remember("name", name)
            return PluginResponse(message=f"Hello, {name}. I'll remember that.")

        remembered = await ctx.recall("name")
        if remembered:
            return PluginResponse(message=f"Hello again, {remembered}.", data={"remembered": True})
        return PluginResponse(
            message="Hello. Tell me your name and I'll remember it.",
            data={"remembered": False},
        )

    async def _ask(self, ctx: PluginContext, question: str) -> PluginResponse:
        """Forward a question to the language model.

        Args:
            ctx: Granted capabilities.
            question: What to ask.

        Returns:
            The model's answer.
        """
        if not question.strip():
            return PluginResponse(message="Ask me something.")
        return PluginResponse(message=await ctx.ask(question))


#: The registry looks for exactly this name.
plugin = HelloPlugin()

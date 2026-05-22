"""Exceptions des guardrails AetherLife."""
from __future__ import annotations


class InvariantViolationError(RuntimeError):
    """Levée quand un invariant runtime est violé."""

    def __init__(self, invariant_id: str, message: str, context: dict | None = None) -> None:
        self.invariant_id = invariant_id
        self.context = context or {}
        full = f"[{invariant_id}] {message}"
        if context:
            full += f" | context={context}"
        super().__init__(full)

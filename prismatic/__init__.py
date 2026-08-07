"""Prismatic/OpenVLA source bundled with an inference-safe lazy public API."""

from importlib import import_module


_MODEL_EXPORTS = {
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
    "load_vla",
}


def __getattr__(name):
    """Load the full upstream model stack only when its public helpers are requested."""
    if name in _MODEL_EXPORTS:
        return getattr(import_module("prismatic.models"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_MODEL_EXPORTS)

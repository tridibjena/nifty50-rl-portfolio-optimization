"""Signal generators, allocators and meta-strategies."""

from .signals import SIGNAL_REGISTRY, make_signal_fn

__all__ = ["SIGNAL_REGISTRY", "make_signal_fn"]

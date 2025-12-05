# src/__init__.py

# Only import the engine, which works without mp_api
from .reaction_engine import ReactionEngine

__all__ = ["ReactionEngine"]
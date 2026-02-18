"""Prompt-injection defense implementations for the RVU framework."""

from rvu.defenses.base import BaseDefense
from rvu.defenses.fath_adapter import FATHDefense
from rvu.defenses.rvg_only import RVGOnlyDefense
from rvu.defenses.vanilla import VanillaDefense

# PromptGuardDefense and RVUDefense require heavy optional dependencies
# (torch, transformers, faiss, sentence-transformers).  Import lazily to
# avoid ImportError when these packages are not installed.
try:
    from rvu.defenses.promptguard_defense import PromptGuardDefense
except ImportError:
    PromptGuardDefense = None  # type: ignore[misc,assignment]

try:
    from rvu.defenses.rvu import RVUDefense
except ImportError:
    RVUDefense = None  # type: ignore[misc,assignment]

try:
    from rvu.defenses.rvu_v2 import RVUv2Defense
except ImportError:
    RVUv2Defense = None  # type: ignore[misc,assignment]

__all__ = [
    "BaseDefense",
    "FATHDefense",
    "PromptGuardDefense",
    "RVGOnlyDefense",
    "RVUDefense",
    "RVUv2Defense",
    "VanillaDefense",
]

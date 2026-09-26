"""Data contracts shared across NS-SER modules (docs/module_interfaces.md)."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConceptToken:
    name: str          # UPPER_SNAKE_CASE, khớp ontology — vd "F0_HIGH"
    confidence: float  # 0.0 - 1.0


@dataclass
class ConceptPrediction:
    audio_id: str
    concepts: List[ConceptToken]


@dataclass
class TraceStep:
    rule_id: str
    premise: List[str]
    conclusion: str
    explanation: str   # lấy trực tiếp từ field explanation trong rule .yaml


@dataclass
class EmotionPrediction:
    audio_id: str
    emotion: Optional[str]  # None nếu reasoning engine không match được emotion nào (xem docs/ontology_design.md)
    confidence: float
    trace: Optional[List[TraceStep]] = None   # None cho 2 baseline
    source: str = "neuro_symbolic"            # "neuro_symbolic" | "end_to_end" | "concept_bottleneck"

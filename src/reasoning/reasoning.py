"""Forward-chaining reasoning engine (docs/ontology_design.md, section 4)."""

from typing import List

from knowledge.kg import KnowledgeGraph
from knowledge.types import ConceptPrediction, EmotionPrediction, TraceStep

# Mỗi dimension chỉ có đúng 2 state đối lập (docs/ontology_design.md mục 1).
_DIMENSION_PAIRS = [
    ("F0_HIGH", "F0_LOW"),
    ("ENERGY_HIGH", "ENERGY_LOW"),
    ("RATE_FAST", "RATE_SLOW"),
    ("PAUSE_LONG", "PAUSE_SHORT"),
]


def infer(concepts: ConceptPrediction, kg: KnowledgeGraph) -> EmotionPrediction:
    """Forward-chain over `kg` from `concepts` to a fixpoint and read off the emotion.

    Facts are binary (concept names only) — ConceptToken.confidence is not
    used to match rules at this stage.
    """
    facts = {token.name for token in concepts.concepts}

    for high, low in _DIMENSION_PAIRS:
        if high in facts and low in facts:
            raise ValueError(
                f"Conflicting concept tokens cho cùng 1 dimension: '{high}' và '{low}' "
                "cùng có mặt trong facts đầu vào."
            )

    trace: List[TraceStep] = []
    fired = set()

    changed = True
    while changed:
        changed = False
        for node, data in kg.nodes(data=True):
            if data.get("kind") != "rule" or node in fired:
                continue

            premise = [u for u, _, d in kg.in_edges(node, data=True) if d.get("relation") == "premise"]
            matched_premise = [p for p in premise if p in facts]

            if data["type"] == "all":
                is_match = set(premise) <= facts
            elif data["type"] == "threshold":
                is_match = len(matched_premise) >= data["min_matches"]
            else:
                is_match = False

            if is_match:
                conclusion = data["conclusion"]
                facts.add(conclusion)
                trace.append(
                    TraceStep(
                        rule_id=node,
                        premise=matched_premise,
                        conclusion=conclusion,
                        explanation=data["explanation"],
                    )
                )
                fired.add(node)
                changed = True

    emotion_facts = [f for f in facts if kg.nodes[f].get("kind") == "emotion"]

    if len(emotion_facts) > 1:
        raise RuntimeError(
            "Nhiều hơn 1 emotion fact được suy ra cùng lúc (rule set đang "
            f"conflict, không silently chọn 1 cái): {emotion_facts}"
        )

    if len(emotion_facts) == 1:
        return EmotionPrediction(
            audio_id=concepts.audio_id,
            emotion=emotion_facts[0],
            confidence=1.0,
            trace=trace,
            source="neuro_symbolic",
        )

    trace.append(
        TraceStep(
            rule_id="none",
            premise=[],
            conclusion="no_match",
            explanation="Không suy luận tiếp được từ facts hiện có.",
        )
    )
    return EmotionPrediction(
        audio_id=concepts.audio_id,
        emotion=None,
        confidence=0.0,
        trace=trace,
        source="neuro_symbolic",
    )

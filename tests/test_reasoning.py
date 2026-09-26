import networkx as nx
import pytest

from knowledge.kg import load_kg
from knowledge.types import ConceptPrediction, ConceptToken
from reasoning.reasoning import infer


def _concepts(*names):
    return ConceptPrediction(
        audio_id="sample_001",
        concepts=[ConceptToken(name=name, confidence=0.9) for name in names],
    )


def test_anger_reasoning():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_HIGH", "RATE_FAST", "PAUSE_SHORT"), kg)

    assert result.emotion == "ANGER"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_HIGH"
    assert result.trace[0].conclusion == "HIGH_AROUSAL"
    assert result.trace[1].rule_id == "R_ANGER"
    assert result.trace[1].conclusion == "ANGER"


def test_happy_reasoning():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_HIGH", "RATE_FAST", "PAUSE_LONG"), kg)

    assert result.emotion == "HAPPY"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_HIGH"
    assert result.trace[0].conclusion == "HIGH_AROUSAL"
    assert result.trace[1].rule_id == "R_HAPPY"
    assert result.trace[1].conclusion == "HAPPY"


def test_sad_reasoning():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_LOW", "ENERGY_LOW", "RATE_SLOW", "PAUSE_LONG"), kg)

    assert result.emotion == "SAD"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_LOW"
    assert result.trace[0].conclusion == "LOW_AROUSAL"
    assert result.trace[1].rule_id == "R_SAD"
    assert result.trace[1].conclusion == "SAD"


def test_neutral_reasoning():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_LOW", "ENERGY_LOW", "RATE_SLOW", "PAUSE_SHORT"), kg)

    assert result.emotion == "NEUTRAL"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_LOW"
    assert result.trace[0].conclusion == "LOW_AROUSAL"
    assert result.trace[1].rule_id == "R_NEUTRAL"
    assert result.trace[1].conclusion == "NEUTRAL"


def test_high_arousal_threshold_2_of_3():
    # Chỉ 2/3 input (thiếu RATE_FAST) vẫn đủ đạt min_matches=2.
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_HIGH", "PAUSE_SHORT"), kg)

    assert result.emotion == "ANGER"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_HIGH"
    assert result.trace[0].premise == ["F0_HIGH", "ENERGY_HIGH"]
    assert result.trace[0].conclusion == "HIGH_AROUSAL"
    assert result.trace[1].rule_id == "R_ANGER"
    assert result.trace[1].conclusion == "ANGER"


def test_high_arousal_1_of_3_fails():
    # Chỉ 1/3 input high, không đủ min_matches=2 -> R_AROUSAL_HIGH không fire.
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "PAUSE_SHORT"), kg)

    assert result.emotion is None
    assert len(result.trace) == 1
    assert result.trace[0].rule_id == "none"
    assert result.trace[0].conclusion == "no_match"


def test_low_arousal_threshold_2_of_3():
    # Chỉ 2/3 input (thiếu RATE_SLOW) vẫn đủ đạt min_matches=2.
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_LOW", "ENERGY_LOW", "PAUSE_LONG"), kg)

    assert result.emotion == "SAD"
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_LOW"
    assert result.trace[0].premise == ["F0_LOW", "ENERGY_LOW"]
    assert result.trace[0].conclusion == "LOW_AROUSAL"
    assert result.trace[1].rule_id == "R_SAD"
    assert result.trace[1].conclusion == "SAD"


def test_low_arousal_1_of_3_fails():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_LOW", "PAUSE_LONG"), kg)

    assert result.emotion is None
    assert len(result.trace) == 1
    assert result.trace[0].rule_id == "none"
    assert result.trace[0].conclusion == "no_match"


def test_no_match_missing_pause():
    """Case biên (robustness), không phải luồng suy luận thường gặp — pipeline
    thực tế luôn trích đủ 4 dimension. Nhánh A trong docs/ontology_design.md
    mục 5: đủ 3/3 token arousal nhưng thiếu PAUSE -> arousal suy ra được
    (1 rule fire) nhưng không rule emotion nào fire tiếp -> no-match được
    APPEND vào trace thật, trace giữ lại bước đã fire trước đó (dài 2 bước).
    """
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_HIGH", "RATE_FAST"), kg)

    assert result.emotion is None
    assert len(result.trace) == 2
    assert result.trace[0].rule_id == "R_AROUSAL_HIGH"
    assert result.trace[0].conclusion == "HIGH_AROUSAL"
    assert result.trace[1].rule_id == "none"
    assert result.trace[1].conclusion == "no_match"


def test_no_match_missing_arousal_dimension():
    """Case biên (robustness), không phải luồng suy luận thường gặp — pipeline
    thực tế luôn trích đủ 4 dimension. Nhánh B trong docs/ontology_design.md
    mục 5: thiếu hẳn 1 trong 3 token arousal (ở đây thiếu RATE) -> theo
    pigeonhole không rule arousal nào đạt min_matches=2 -> không rule nào
    fire trong toàn bộ fixpoint -> trace chỉ có đúng bước no-match (1 bước).
    """
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_LOW", "PAUSE_SHORT"), kg)

    assert result.emotion is None
    assert len(result.trace) == 1
    assert result.trace[0].conclusion == "no_match"


def test_rule_does_not_fire_twice():
    kg = load_kg("knowledge")
    result = infer(_concepts("F0_HIGH", "ENERGY_HIGH", "RATE_FAST", "PAUSE_SHORT"), kg)

    rule_ids = [step.rule_id for step in result.trace]
    assert len(rule_ids) == len(set(rule_ids))
    assert rule_ids == ["R_AROUSAL_HIGH", "R_ANGER"]


def test_conflicting_input_tokens_raises():
    # F0_HIGH và F0_LOW cùng thuộc dimension F0 -> vi phạm contract
    # "mỗi dimension emit đúng 1 token" (docs/ontology_design.md mục 1).
    kg = load_kg("knowledge")
    with pytest.raises(ValueError, match="F0"):
        infer(_concepts("F0_HIGH", "F0_LOW", "ENERGY_HIGH", "RATE_FAST", "PAUSE_SHORT"), kg)


def test_emotion_conflict_raises():
    """KG test-only (không đụng knowledge/ thật): 2 rule cùng premise "A",
    mỗi rule conclude 1 emotion khác nhau -> 1 facts set kích hoạt cả 2 cùng
    lúc -> vi phạm cardinality "đúng 1 emotion fact" (mục 4) -> RuntimeError.
    """
    kg = nx.DiGraph()
    kg.add_node("A", kind="acoustic_concept")
    kg.add_node("E1", kind="emotion")
    kg.add_node("E2", kind="emotion")
    kg.add_node("R1", kind="rule", type="all", min_matches=None, conclusion="E1", explanation="test rule 1")
    kg.add_node("R2", kind="rule", type="all", min_matches=None, conclusion="E2", explanation="test rule 2")
    kg.add_edge("A", "R1", relation="premise")
    kg.add_edge("R1", "E1", relation="conclusion")
    kg.add_edge("A", "R2", relation="premise")
    kg.add_edge("R2", "E2", relation="conclusion")

    concepts = ConceptPrediction(audio_id="sample_conflict", concepts=[ConceptToken(name="A", confidence=0.9)])

    with pytest.raises(RuntimeError):
        infer(concepts, kg)

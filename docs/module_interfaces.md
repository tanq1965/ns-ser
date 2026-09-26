## Data storage structure + Module interfaces

### Cấu trúc lưu trữ dữ liệu
```
data/
  raw/                        # audio gốc MELD/IEMOCAP
  processed/
    features/                 # đặc trưng âm học đã trích (.npy / .parquet)
    concepts/                 # concept token dự đoán/nhãn (.json / .csv)
  DATA_CARD.md

knowledge/
  ontology.yaml                # (docs/ontology_design.md)
  rules/
    *.yaml

experiments/
  configs/
    *.yaml
  runs/
    <run_id>/
      metrics.json
      trace_samples.json
      checkpoint/               # KHÔNG commit
```

### Module interfaces — data contract
```python
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
```

### Function-level contract (chưa implement)
```python
# src/perception/
def extract_features(audio_path: str) -> "AcousticFeatures": ...
def predict_concepts(features: "AcousticFeatures") -> ConceptPrediction: ...

# src/knowledge/
def load_kg(path: str) -> "KnowledgeGraph": ...

# src/reasoning/
def infer(concepts: ConceptPrediction, kg: "KnowledgeGraph") -> EmotionPrediction: ...

# src/models/  — 2 baseline, PHẢI trả cùng schema EmotionPrediction
def end_to_end_predict(audio_path: str) -> EmotionPrediction: ...                     # trace=None
def concept_bottleneck_predict(concepts: ConceptPrediction) -> EmotionPrediction: ...  # trace=None

# src/pipeline/
def run(audio_path: str, mode: str) -> EmotionPrediction: ...
# mode: "neuro_symbolic" | "end_to_end" | "concept_bottleneck"

# src/evaluation/
def compare(predictions: List[EmotionPrediction], ground_truth: dict) -> "Report": ...
```

Điểm thiết kế: `trace=None` ở 2 baseline là cách evaluation module tính rule coverage bằng cách lọc `trace is not None`.
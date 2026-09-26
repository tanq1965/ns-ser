## Ontology + KG + Rule set + Reasoning engine

### 1. Ontology

3 lớp, quan hệ 2 tầng (concept → arousal → emotion), giữ đúng nguyên tắc "4 concept token cố định":

```yaml
# knowledge/ontology.yaml
acoustic_concepts:
  - F0_HIGH
  - F0_LOW
  - ENERGY_HIGH
  - ENERGY_LOW
  - RATE_FAST
  - RATE_SLOW
  - PAUSE_LONG
  - PAUSE_SHORT
intermediate_concepts:
  - HIGH_AROUSAL
  - LOW_AROUSAL
emotions:
  - ANGER
  - HAPPY
  - SAD
  - NEUTRAL
```

**Quyết định: mỗi dimension chỉ 2 state (HIGH/LOW hoặc FAST/SLOW, LONG/SHORT), median split trên tập train.** Không dùng 3 mức (có NEUTRAL band) — giữ rule set nhỏ nhất có thể, đúng nguyên tắc "kích thước kiểm soát được".

**Giới hạn đã biết:** median tính trên toàn bộ tập train (gộp nhiều speaker) không đảm bảo cùng ý nghĩa giữa các speaker có tầm vực giọng khác nhau — VD 1 speaker trầm có thể gần như toàn bộ utterance rơi vào LOW dù không liên quan tới arousal thấp. IEMOCAP split theo Session (đã chốt ở D3) giảm phần nào rủi ro train/test leakage nhưng không loại bỏ hoàn toàn nguy cơ này. Không xử lý per-speaker normalization ở Phase 1 (ngoài scope) — ghi nhận là giới hạn cần nêu ở error analysis Tuần 10-11.

**Contract cho src/perception/:** mỗi dimension phải emit đúng 1 token, không 0 không 2: F0 ∈ {F0_HIGH, F0_LOW}, ENERGY ∈ {ENERGY_HIGH, ENERGY_LOW}, RATE ∈ {RATE_FAST, RATE_SLOW}, PAUSE ∈ {PAUSE_LONG, PAUSE_SHORT}. reasoning engine validate điều này và raise ValueError nếu vi phạm (xem mục 4).

**Chỉ 1 intermediate concept: AROUSAL** (không làm VALENCE riêng). Lý do: tương quan F0/energy/rate với arousal là finding được lặp lại nhiều nhất trong tài liệu vocal-affect (Scherer 2003); Valence được loại khỏi ontology ở giai đoạn này vì tín hiệu prosody đơn thuần cung cấp bằng chứng hạn chế và phụ thuộc ngữ cảnh cho suy luận valence đáng tin cậy — đây là quyết định về phạm vi (scope limitation), không phải kết luận rằng valence không thể suy luận được từ giọng nói. PAUSE dùng làm tie-break trực tiếp ở tầng emotion thay vì làm intermediate concept riêng, để không overclaim cơ sở lý thuyết cho nó.

**Lưu ý về semantics:** mỗi acoustic concept là một *discretized feature* (raw feature → median-split trên tập train → binary token), không phải nhãn có ý nghĩa độc lập. "F0_HIGH" nghĩa là "F0 cao hơn median của tập train", không phải ngưỡng sinh lý/âm học tuyệt đối — pipeline đầy đủ: `raw feature → threshold (median, fit trên train, freeze) → concept token → symbolic reasoning`.

### 2. Quyết định biểu diễn KG: **networkx, không dùng rdflib/.ttl**

Lý do: không cần OWL reasoner hay SPARQL — reasoning engine tự viết forward-chaining, rdflib chỉ thêm dependency/cú pháp Turtle không cần thiết cho KG ~14 node. `knowledge/ontology.ttl` trong D2 đổi thành `knowledge/ontology.yaml` (ở trên).

**Biểu diễn cụ thể: bipartite fact-rule graph** (giống tinh thần Rete network) — mỗi rule là 1 node riêng, không phải edge trực tiếp giữa 2 fact, vì có rule cần 3 input (threshold "≥2/3") mà edge thường (1→1) không biểu diễn được:

```
concept nodes ──premise──▶ rule node ──conclusion──▶ concept/intermediate/emotion node
```

`load_kg()` dựng graph này 1 lần từ `ontology.yaml` + `rules/*.yaml` lúc khởi động, không cần persist ra file KG riêng.

### 3. Rule set tối thiểu (6 rule, phủ đầy đủ 2×2×2 → 4 lớp)

```yaml
# knowledge/rules/arousal.yaml
- id: R_AROUSAL_HIGH
  type: threshold
  inputs: [F0_HIGH, ENERGY_HIGH, RATE_FAST]
  min_matches: 2
  conclusion: HIGH_AROUSAL
  explanation: "Khi ≥2/3 đặc trưng (F0, energy, tốc độ nói) ở mức cao, suy ra arousal cao — tương quan prosody-arousal được ghi nhận rộng rãi (Scherer, 2003)."

- id: R_AROUSAL_LOW
  type: threshold
  inputs: [F0_LOW, ENERGY_LOW, RATE_SLOW]
  min_matches: 2
  conclusion: LOW_AROUSAL
  explanation: "Khi ≥2/3 đặc trưng ở mức thấp, suy ra arousal thấp."
```

```yaml
# knowledge/rules/emotion.yaml
- id: R_ANGER
  type: all
  inputs: [HIGH_AROUSAL, PAUSE_SHORT]
  conclusion: ANGER
  explanation: "Arousal cao + ít ngắt quãng — khớp mô tả vocal cues của anger (nói dồn dập, ít pause) trong Scherer (2003)."

- id: R_HAPPY
  type: all
  inputs: [HIGH_AROUSAL, PAUSE_LONG]
  conclusion: HAPPY
  explanation: "Heuristic yếu nhất trong rule set — happy không có tương quan pause rõ ràng như anger trong tài liệu; dùng làm nhánh còn lại để rule set đầy đủ. Dự kiến là nguồn lỗi chính, cần xác nhận ở error analysis Tuần 10-11."

- id: R_SAD
  type: all
  inputs: [LOW_AROUSAL, PAUSE_LONG]
  conclusion: SAD
  explanation: "Arousal thấp + nhiều khoảng lặng — khớp mô tả vocal cues của sadness (chậm, trầm, ngắt quãng nhiều)."

- id: R_NEUTRAL
  type: all
  inputs: [LOW_AROUSAL, PAUSE_SHORT]
  conclusion: NEUTRAL
  explanation: "Arousal thấp, không ngắt quãng bất thường — dùng làm nhánh mặc định trong không gian low-arousal."
  note: "Nhánh mặc định trong không gian low-arousal, KHÔNG phải bằng chứng dương tính mạnh cho 'neutral' cụ thể — một utterance SAD nhưng ít biến thiên prosody (nói đều, ít pause) có thể bị gán nhầm NEUTRAL. Giới hạn cần theo dõi ở error analysis, tương tự R_HAPPY."
```

Rule set tạo một exhaustive partition trên state space nhị phân (AROUSAL × PAUSE = 4 tổ hợp, mỗi tổ hợp có đúng 1 rule match) — đây là tính chất của logical coverage, KHÔNG phải claim rằng 4 emotion class là mutually exclusive về mặt ground-truth/semantic (một audio thực tế có thể mang cues của nhiều cảm xúc cùng lúc).

### 4. Reasoning engine — thuật toán forward-chaining

Khớp đúng signature D2 đã chốt (`infer(concepts, kg) -> EmotionPrediction`):

```
facts = {tên các concept token trong ConceptPrediction}      # vd {F0_HIGH, ENERGY_LOW, RATE_FAST, PAUSE_SHORT}
trace = []
lặp đến khi không còn rule mới match (fixpoint, KHÔNG hardcode số bước):
    với mỗi rule chưa fire:
        nếu type="all": match khi mọi input ⊆ facts
        nếu type="threshold": match khi |inputs ∩ facts| ≥ min_matches
        nếu match: thêm conclusion vào facts, ghi TraceStep(rule.id, inputs khớp, conclusion, rule.explanation)
emotion = fact thuộc lớp Emotion trong facts   # đúng 1 với rule set này
```

**Cardinality của emotion output:** 0 fact thuộc kind=emotion → no-match (mục 5-6). Đúng 1 → trả kết quả bình thường. NHIỀU HƠN 1 (không thể xảy ra với 6 rule hiện tại vì mỗi tổ hợp AROUSAL×PAUSE chỉ map 1 rule — nhưng có thể xảy ra khi mở rộng rule set sau này) → raise RuntimeError, KHÔNG silently chọn 1 cái. Engine phải fail loud khi rule set được mở rộng theo cách tạo conflict, không âm thầm trả kết quả sai.

**Quyết định phụ:**
- `confidence` trong `ConceptToken` KHÔNG dùng để match rule ở giai đoạn này (chỉ facts nhị phân) — giữ đơn giản, có thể nâng cấp sau.
- Nếu hết fixpoint mà không có Emotion nào trong facts (không xảy ra với rule set hiện tại, nhưng cần cho robustness): `EmotionPrediction(emotion=None, trace=[TraceStep(..., conclusion="no_match")])` — evaluation module dùng chính flag này để tính rule coverage, khớp thiết kế `trace is not None` đã chốt ở D2.
- Vòng lặp fixpoint (không phải depth cố định) để dễ mở rộng khi thêm rule sau này (VD lên 6-class) mà không cần sửa engine.

Phần này đủ chi tiết để đưa cho Claude Code. Bạn đã sẵn sàng chuyển sang phần của Trí (perception + baseline neural) luôn, hay muốn mình rà lại điểm nào trong 4 mục trên trước?

### 5. Semantics của no-match

**Nguyên lý (pigeonhole):** `R_AROUSAL_HIGH`/`R_AROUSAL_LOW` dùng `min_matches=2` trên 3 input nhị phân (F0/ENERGY/RATE). Nếu cả 3 dimension đều có token — bất kể giá trị gì — tổng luôn bằng 3, nên một phía HIGH/LOW chắc chắn đạt ≥2. Do đó no-match ở tầng AROUSAL **chỉ xảy ra khi thiếu hẳn ≥1 trong 3 token F0/ENERGY/RATE**, không bao giờ do "tín hiệu lẫn lộn" khi đủ cả 3 — pipeline thực tế luôn trích đủ 4 dimension nên đây là case biên (robustness), không phải luồng suy luận thường gặp.

no-match có 2 nhánh, khác nhau về độ dài trace vì engine **append** vào trace thật thay vì thay thế (xem mục 4):

**Nhánh A — đủ 3 token arousal, thiếu PAUSE.** Arousal suy ra được (1 rule fire), nhưng không rule emotion nào match tiếp vì thiếu PAUSE_LONG/PAUSE_SHORT. Trace dài 2 bước.

```
facts = {F0_HIGH, ENERGY_HIGH, RATE_FAST}
→ R_AROUSAL_HIGH fires → HIGH_AROUSAL
→ không rule emotion nào fire (thiếu PAUSE)
→ trace = [R_AROUSAL_HIGH→HIGH_AROUSAL, none→no_match], emotion=None
```

**Nhánh B — thiếu 1 trong 3 token arousal.** Không rule arousal nào đạt `min_matches=2`, nên không có rule nào fire trong toàn bộ fixpoint. Trace dài 1 bước.

```
facts = {F0_HIGH, ENERGY_LOW, PAUSE_SHORT} # RATE token vắng mặt
→ R_AROUSAL_HIGH: chỉ khớp F0_HIGH (1/3) < 2 → không fire
→ R_AROUSAL_LOW: chỉ khớp ENERGY_LOW (1/3) < 2 → không fire
→ trace = [none→no_match], emotion=None
```

Với rule set hiện tại, MỘT KHI arousal đã suy ra được VÀ có PAUSE concept, emotion luôn suy ra được (4 tổ hợp AROUSAL×PAUSE cover đủ). Sơ đồ:

```
Acoustic concepts (đủ 4 dimension)
│
▼
Arousal inference
├─ matched (nhánh A nếu thiếu PAUSE) ──▶ có PAUSE? ─┬─ có ──▶ Emotion
│ └─ không ─▶ no-match (2-step trace)
└─ insufficient evidence (thiếu token, nhánh B) ──────────────▶ no-match (1-step trace)
```

### 6. Hành vi trace khi no-match

no-match APPEND 1 `TraceStep(rule_id="none", conclusion="no_match", ...)` vào trace đã ghi được, KHÔNG thay thế toàn bộ trace. Lý do: nếu 1 rule đã thực sự fire trước khi tắc (như nhánh A ở mục 5), xoá mất bước đó sẽ làm mất thông tin suy luận đã đi được tới đâu — ngược với mục tiêu explainability của reasoning trace.
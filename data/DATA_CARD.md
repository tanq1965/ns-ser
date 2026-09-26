# Data Card — NS-SER

## Dataset chính: IEMOCAP
- Trạng thái access: chưa có (yêu cầu form license ký với USC SAIL, không phân phối tự do)
- Class subset: 4-class (angry, happy, sad, neutral) — khớp nguyên tắc   "nhỏ chạy được end-to-end"
- Split: theo Session (Session 1-4 train, Session 5 test) — tránh speaker leakage

## Dataset dự phòng: MELD
- Dùng khi IEMOCAP access bị trễ quá thời hạn cho phép trong kế hoạch 14 tuần
- Split strategy: CHƯA CHỐT

## Concept labeling
- Phương pháp: rule-threshold — trích F0/energy/rate/pause thô, áp ngưỡng median-split tính trên tập train, gán concept token trực tiếp
- KHÔNG train MLP ở giai đoạn này
- Ngưỡng fit trên train, freeze cho val/test — không tính lại trên val/test (tránh leakage)
- Ngưỡng KHÔNG mang được giữa các dataset khác nhau (IEMOCAP/MELD/RAVDESS) — phải tính lại hoàn toàn nếu đổi dataset

## Lưu trữ
- Audio gốc và feature đã trích KHÔNG commit vào git (data/raw/, data/processed/ trong .gitignore) — license của IEMOCAP/MELD không cho phép phân phối lại
- Cấu trúc lưu trữ đầy đủ: xem docs/module_interfaces.md
# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab: GPU** CPU/GPU

**Python / PyTorch / Ultralytics: 3.13.16 / 2.11.0+cu130 / 8.4.145**

**Checkpoint: yolo11n-cls.pt sha256=c62d41bf9625…, yolo11n.pt sha256=0ebbc80d4a76…, yolo11n-seg.pt sha256=55ed65c56c91… ** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON: 0.35** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn: Không** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
    [{
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K",
    },
    ]

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Record nói về cả bức ảnh. Vì nó cho thấy id của ảnh, task, rank, width, height của cả bức ảnh đó.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp không từ model nghĩ ra mà được xác định bởi taxonomy/checkpoint.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Vì mỗi trường có vai trò khác nhau trong việc quản lý và phân loại dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần nói rõ mình cần đưa rõ nguyên tắc chọn lớp đại diện để người gán nhãn đưa ra một kết quả nhất quán.
6. Vì sao không được lấy `score` của model làm ground truth?
    Độ tin cậy của chất lượng không phải là ground truth vì score thể hiện độ tin cậy của model đối với dự đoán chứ không chứng minh được dự đoán đó đúng.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
     {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Góc trên-trái ở vị trí: (385.33, 69.24); Góc dưới-phải: (498.92, 348.92); Rộng: 113.58: Cao: 279.68.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    Ngưỡng 0.35 có 11 vật thể, 
    Ngưỡng 0.60 có 6 vật thể,
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Người reviewer phải kiểm tra hộp sai, kiểm tra hộp trùng lặp, kiểm tra class, kiểm tra vị trí bounding box, kiểm tra đối tượng bị sót.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Box bao phủ toàn bộ phần đối tượng nhìn thấy được.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Bow che khuất một phần của bàn. Bạn nên đưa lên hỏi lab coach, vì chưa hiểu rõ đối tượng bị che khuất, và có thể ảnh hưởng đến tính nhất quán của dữ liệu.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-007",
    "class_id": 69,
    "class_name": "oven",
    "score": 0.489667,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      0.25,
      188.95,
      194.35,
      294.62
    ],
    "polygon_point_count": 213,
    "polygon_xy": [
      [
        1.0,
        189.0
      ],
      [
        1.0,
        218.0
      ],
      [
        1.0,
        214.0
      ],
      ]
    }
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    So với box, polygon cho biết thêm thông tin về đường biên, diện tích thực, độ che khuất và hình dạng của vật thể. 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    instance_id dùng để định danh một đối tượng cụ thể trong ảnh. Trong ảnh có 2 con sư tử cùng thuộc class là lion, thì mỗi con có một instance_id riêng, ảnh tĩnh thì không cần phải tracking thì không cần tracking ID
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    người gán nhãn phải vẽ đúng đường biên thực tế của đối tượng.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Trên ảnh kichen, bow 0.74 che bow 0.55. Trước tiên thì dùng guideline. Chỉ escalation ghi gặp trường hợp chưa được quy tắc hoặc không thể áp dụng quy tắc đáng tin cậy hiện có.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class_id, class_name | Ảnh có nhiều vật thể khó xác định chủ thể chính | Xác định chủ thể chính theo guideline, chọn đúng 1 lớp  | class_id và name có khớp với taxonomy không.  |
| Phát hiện vật thể | bbox_format, bbox_xyxy,bbox_width,bbox_height | vật thể bị chồng nhau | vẽ bounding box theo guideline, không xác định được báo lab coach | Kiểm tra box có bao trọn vật thể không |
| Instance segmentation | polygon_point_count, polygon_xy | vật thể bị che khuất, biên vật thể mờ, không bao hết vật thể | vẽ theo đúng đường biên | kiểm tra mask có bám sát biên không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
     chỉ sử dụng ảnh có sẵn không dùng ảnh cá nhân hay dữ liệu nội bộ.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Sẽ dừng vào báo cáo cho lab coach.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

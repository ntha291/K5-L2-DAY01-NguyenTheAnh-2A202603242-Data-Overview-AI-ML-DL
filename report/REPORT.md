# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

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

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này đưa ra dự đoán về toàn về nội dung bức ảnh vì nó liệt kê ra từng vật thể có trong bức ảnh này


3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Đi kèm taxonomy và checkpoint vì danh sách tên lớp không bao giờ do mô hình tự nghĩ ra tại thời điểm suy luận

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì class_id giúp khoá ổn định cho máy, class_name để người đọc hiểu và phát hiện map sai, taxonomy_name cho biết id thuộc bộ nhãn nào, vì cùng một id có thể có nghĩa khác nhau ở taxonomy khác
Nếu chỉ ghi mỗi tên lớp, ta dễ gặp trùng nghĩa và không truy vết được khi taxonomy thay đổi

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Quy tắc chọn 1 lớp: ưu tiên chọn chủ thể chính (vật lớn nhất, nằm ở trung tâm,...), thứ tự ưu tiên giữa các lớp, cách xử lý khi hoà, ví dụ cho các ca biên

6. Vì sao không được lấy `score` của model làm ground truth?
‘score’ chỉ là độ tự tin của model, không phải sự thật. Model có thể tự tin mà vẫn sai. Dùng nó làm nhãn rồi đánh giá chính model đó sẽ bị tự xác nhận, nên metric đẹp giả. Chỉ nên dùng ‘score’ để hỗ trợ review, còn ground truth phải do người gán. 

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=’person’ ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 384.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Góc trên-trái: (385.33, 69.24)
Góc dưới-phải: (498.92, 348.92)
Rộng: 498.92 − 385.33 = 113.58 px
Cao: 348.92 − 69.24 = 279.68 px


3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
0.20 có 53 prediction
0.60 có 25 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Phải duyệt nhiều box hơn, và các box này có score thấp nên sai nhiều hơn. Cụ thể là loại box sai, sửa nhãn sai, chỉnh box lệch, gộp box trùng. Mỗi quyết định giữ, sửa hay xóa cũng cần được ghi lại. 

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Mỗi vật thật có một box, ôm sát phần nhìn thấy của vật: các cạnh box nằm trên đường biên của vật, không chừa nhiều nền, không cắt mất phần vật. 

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Box person ở mép trái chỉ là bàn tay và cẳng tay đưa vào khung hình, ngay cạnh các đĩa bánh. Phần còn lại của người đó nằm ngoài ảnh. Vì vậy đây vừa là vật bị cắt mép, vừa có thêm câu hỏi: chỉ thấy một bộ phận cơ thể thì có tính là person không?
Tôi tách làm hai phần:
Viết vào guideline quy tắc cho vật bị cắt mép: box ôm phần nhìn thấy, cạnh box trùng mép ảnh. Quy tắc này cũng áp dụng cho dĩa bánh ở góc dưới-trái (bowl, score 0.38) và lò nấu ở mép trái.
Hỏi Lab Coach câu “chỉ thấy bàn tay thì có gán person không?”. Guideline chưa có quy tắc về mức độ nhìn thấy tối thiểu, nên tôi không tự quyết được. Khi có câu trả lời thì ghi vào guideline để lần sau không cần hỏi nữa.




## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0] 

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho hình dạng thật của vật: chỗ lõm, khoảng hở giữa tay, thân và chân, và phần nền nằm trong box. Với người trong ảnh này, polygon chỉ chiếm khoảng 18.170 px², trong khi box rộng 31.760 px², tức là khoảng 43% diện tích box là nền. Box thì không nói được điều đó. 

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Nó là tên riêng của từng vật cụ thể trong một ảnh, ví dụ kitchen-001 là người, kitchen-002 là một cái bát. Nhờ đó mỗi mask, mỗi lần review và mỗi lần sửa gắn đúng vào một vật.
Không phải class ID: hai cái bát có cùng class_id nhưng khác instance_id.
Không phải tracking ID: nó chỉ có nghĩa trong một ảnh, không nói vật này là vật nào ở ảnh khác hay khung hình khác.


4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Biên mask đi theo đường viền thật của phần nhìn thấy của vật, mỗi pixel chỉ thuộc một instance. Chỗ hai vật chạm hoặc chồng nhau thì pixel thuộc về vật nằm phía trước. 

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Bàn tay ở mép trái (kitchen-009, nhãn person) chạm vào cái bát ngay bên dưới (kitchen-006, nhãn bowl): hai box chồng nhau ở khoảng y ≈ 303–310. Đây cũng là chỗ khó vì bàn tay và bát cùng nằm sát mép ảnh.
Tôi dùng guideline cho đường biên giữa hai vật. Chạm và chồng nhau là tình huống lặp lại ở rất nhiều ảnh, và quy tắc ở câu 4 đã đủ để người gán nhãn tự quyết, nên không cần hỏi từng ca.
Câu hỏi “chỉ thấy bàn tay thì có gán person không” vẫn cần escalate lên Lab Coach như đã nói ở lượt trước, vì guideline chưa có quy tắc về mức độ nhìn thấy tối thiểu. Riêng phần biên thì dùng guideline được.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn cho cả ảnh: class_id + class_name + taxonomy_name.  |Ảnh kitchen có người, bát, lò, chảo, cây, nên khó chọn một lớp. Cần quy tắc chủ thể chính hoặc nhãn ambiguous.  | Chọn một lớp theo guideline, ca mơ hồ thì gắn cờ hoặc hỏi. | Nhãn có đúng taxonomy và đúng quy tắc chủ thể chính không, các ca mơ hồ xử lý thống nhất chưa. |
| Phát hiện vật thể | Một box cho mỗi vật: bbox_xyxy (pixel) + class_name. | person ở mép trái ảnh kitchen (x_min = 0.12, score 0.61) chỉ là bàn tay, bị cắt mép. bowl score 0.38 cũng chạm mép trái. Trong file có 28/53 box nằm ở score 0.35–0.60, vùng dễ sai. | Vẽ một box/vật, ôm sát phần nhìn thấy, cạnh box trùng mép ảnh khi vật bị cắt. Hỏi Lab Coach vụ bàn tay có tính là person không. | Box có lệch, thừa, thiếu không, nhãn có đúng không, có box trùng không, các box score thấp có đáng giữ không. |
| Instance segmentation | Một mask/polygon cho mỗi vật: instance_id + class_name + polygon_xy. | Bàn tay (kitchen-009) chạm cái bát (kitchen-006) quanh y ≈ 303–310. Polygon người kitchen-001 chỉ bằng 57% diện tích box (18.170 so với 31.760 px²), nên box không phản ánh đúng hình dạng. | Vẽ biên theo viền thật của phần nhìn thấy, mỗi pixel thuộc một instance, vật nằm trước thắng ở chỗ chồng nhau. | Biên có bám đúng viền không, mask có chồng nhau hoặc lẹm nền không, mỗi instance_id có đúng một vật không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng dữ liệu được cấp cho bài lab này, không chia sẻ ra ngoài hay đưa lên dịch vụ bên thứ ba, và không lưu hay công bố thông tin nhận dạng cá nhân. 

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng lại, không xử lý hay gán nhãn tiếp, rồi báo cho giảng viên hoặc người phụ trách bài lab, kèm tên file hoặc ID của mẫu đó. 

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
# Phân tích dữ liệu giao thông

Ngày phân tích: 22/09/2026. Nguồn: `runs/data.csv` (7.044 bản ghi tại thời điểm phân tích). Công thức chỉ số được đối chiếu với `core/Yolo_detect.py`; quy trình thu thập được tham khảo từ `get_data.py`.

## 1. Kết quả chính

- Dữ liệu bao phủ gần 24 giờ, từ 17:41:40 ngày 21/09/2026 đến 17:41:32 ngày 22/09/2026.
- Số phương tiện trung bình là 22,37 xe/khung hình; trung vị 23 xe. Số xe giảm mạnh ban đêm và tăng rõ từ khoảng 06:00.
- Khung giờ đầy đủ có số xe trung bình cao nhất là 08:00–09:00 ngày 22/09, đạt 39,00 xe/khung hình. Thấp nhất là 02:00–03:00 cùng ngày, đạt 2,68 xe/khung hình.
- Đỉnh tức thời là 59 xe lúc 09:20:57 ngày 22/09/2026.
- Không phát hiện ô thiếu, dòng trùng hoàn toàn, timestamp trùng, số đếm âm hoặc sai lệch công thức `total`, `WTI`, `WTI_norm`.
- `WTI_norm` không phải chỉ số chuẩn hóa về 0–1 hay 0–100. Công thức hiện tại khuếch đại WTI theo tổng số xe.
- Một ngày quan sát đủ cho phân tích mô tả ban đầu, nhưng chưa đủ để xác định quy luật theo ngày trong tuần, xây dựng đường cơ sở ổn định hoặc xác nhận ùn tắc thực tế.

## 2. Phạm vi và cách phân tích

Phân tích đọc toàn bộ CSV, chuyển `timestamp` thành thời gian và kiểm tra kiểu dữ liệu, giá trị thiếu, trùng lặp, miền giá trị, thứ tự thời gian và khoảng cách giữa các mẫu. Các cột dẫn xuất được tính lại từ số đếm phương tiện để đối chiếu.

Thống kê theo giờ được nhóm theo ngày và giờ thực tế, không gộp 17:00 của hai ngày. Trung bình, phân vị và tương quan được tính theo bản ghi; trung bình theo giờ không được gia quyền theo thời lượng giữa các mẫu. Tỷ trọng phương tiện là tổng số lượt phát hiện của từng loại chia cho tổng `total` trên toàn bộ dữ liệu.

Các giá trị trình bày được làm tròn; việc kiểm tra công thức thực hiện trên dữ liệu chưa làm tròn. Tương quan sử dụng hệ số Pearson. Các phân vị sử dụng cách nội suy tuyến tính mặc định của pandas.

Timestamp trong CSV không có thông tin múi giờ. Báo cáo giữ nguyên thời gian ghi trong file, không chuyển đổi múi giờ. Nội dung mô tả vị trí camera trong README không thay thế được trường định danh camera: CSV hiện không có `camera_id` để tự xác minh nguồn từng bản ghi.

## 3. Cấu trúc dữ liệu

| Cột | Ý nghĩa |
|---|---|
| `timestamp` | Thời điểm được ghi nhận, định dạng `YYYY-MM-DD HH:MM:SS` |
| `person` | Số đối tượng được nhận diện là người |
| `car` | Số ô tô được nhận diện |
| `motorbike` | Số xe máy được nhận diện |
| `bus` | Số xe buýt được nhận diện |
| `truck` | Số xe tải được nhận diện |
| `total` | Tổng `car + motorbike + bus + truck`, không gồm `person` |
| `WTI` | Tổng số phương tiện có trọng số theo loại xe |
| `WTI_norm` | WTI nhân với hệ số phụ thuộc logarit của tổng số xe |

Các số đếm biểu thị đối tượng trong từng ảnh. Cùng một phương tiện có thể xuất hiện ở nhiều ảnh, vì vậy không cộng các mẫu để diễn giải thành số phương tiện duy nhất hoặc lưu lượng xe đi qua giao lộ.

## 4. Chất lượng dữ liệu và độ liên tục

| Kiểm tra | Kết quả |
|---|---:|
| Số dòng dữ liệu | 7.044 |
| Số cột | 9 |
| Ô thiếu | 0 |
| Timestamp không phân tích được | 0 |
| Timestamp trùng | 0 |
| Dòng trùng hoàn toàn | 0 |
| Giá trị âm trong các cột số | 0 |
| Số đếm không nguyên | 0 |
| Dòng sai công thức `total` | 0 |
| Dòng sai công thức `WTI` | 0 |
| Dòng không khớp công thức `WTI_norm` trong dung sai số thực | 0 |

Timestamp tăng đúng thứ tự. Thời gian bao phủ là 23 giờ 59 phút 52 giây, gồm 7.043 khoảng cách giữa hai bản ghi liên tiếp.

| Khoảng cách giữa hai mẫu | Giây |
|---|---:|
| Trung bình | 12,27 |
| Độ lệch chuẩn | 2,50 |
| Nhỏ nhất | 3 |
| Phân vị 25% | 12 |
| Trung vị | 12 |
| Phân vị 75% | 12 |
| Lớn nhất | 147 |

Hai khoảng gián đoạn dài hơn 60 giây:

| Mẫu trước | Mẫu sau | Khoảng cách |
|---|---|---:|
| 22/09/2026 10:20:34 | 22/09/2026 10:22:50 | 136 giây |
| 22/09/2026 16:00:08 | 22/09/2026 16:02:35 | 147 giây |

CSV không đủ thông tin để xác định nguyên nhân gián đoạn. Mã thu thập hiện tại bỏ qua ảnh giống ảnh trước và chỉ ghi sau khi xử lý ảnh mới, nên nhịp lấy mẫu không nhất thiết cố định. Không nên coi mọi khoảng trống là đường không có xe.

Có 267 mẫu ghi nhận `total = 0`, tương đương 3,79%. Cần đối chiếu ảnh để phân biệt đường vắng với lỗi nhận diện hoặc ảnh không sử dụng được. Có 66 cặp mẫu liên tiếp giống nhau ở toàn bộ năm cột đếm `person`, `car`, `motorbike`, `bus`, `truck`; số đếm giống nhau không chứng minh ảnh bị trùng và không phải lý do đủ để xóa mẫu.

Không có ô thiếu trong CSV cũng không có nghĩa quá trình thu thập không bỏ lỡ khung hình. Tính nhất quán số học không chứng minh độ chính xác của YOLO.

## 5. Thống kê mô tả

Đơn vị các cột đếm là đối tượng/khung hình. WTI và WTI_norm là chỉ số theo công thức của dự án, chưa phải đơn vị giao thông đã được hiệu chuẩn.

| Biến | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P25 | Trung vị | P75 | Lớn nhất |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 6,81 | 6,64 | 0 | 1 | 5 | 12 | 34 |
| car | 11,77 | 8,22 | 0 | 4 | 11 | 18 | 39 |
| motorbike | 7,32 | 8,24 | 0 | 0 | 4 | 12 | 41 |
| bus | 1,00 | 1,49 | 0 | 0 | 0 | 2 | 14 |
| truck | 2,28 | 2,73 | 0 | 0 | 1 | 3 | 20 |
| total | 22,37 | 15,38 | 0 | 7 | 23 | 36 | 59 |
| WTI | 75,08 | 49,57 | 0 | 30 | 77 | 114 | 267 |
| WTI_norm | 329,01 | 245,85 | 0 | 92,72 | 321,54 | 526,28 | 1.327,07 |

Khoảng 50% số mẫu có 7–36 xe. Độ lệch chuẩn lớn phản ánh sự khác biệt giữa các thời điểm trong ngày cùng dao động giữa các ảnh; riêng thống kê này chưa cho biết nguyên nhân là chuyển động giao thông, đèn tín hiệu hay sai số phát hiện.

## 6. Diễn biến theo giờ

| Ngày | Giờ bắt đầu | Số mẫu | Total trung bình | Total lớn nhất | WTI trung bình |
|---|---|---:|---:|---:|---:|
| 21/09/2026 | 17:00* | 91 | 31,22 | 54 | 104,56 |
| 21/09/2026 | 18:00 | 295 | 26,48 | 48 | 96,01 |
| 21/09/2026 | 19:00 | 295 | 16,74 | 39 | 60,35 |
| 21/09/2026 | 20:00 | 294 | 14,39 | 33 | 50,78 |
| 21/09/2026 | 21:00 | 293 | 13,09 | 32 | 46,80 |
| 21/09/2026 | 22:00 | 295 | 10,82 | 26 | 43,49 |
| 21/09/2026 | 23:00 | 293 | 8,46 | 22 | 35,49 |
| 22/09/2026 | 00:00 | 296 | 6,25 | 24 | 26,54 |
| 22/09/2026 | 01:00 | 293 | 3,99 | 17 | 16,99 |
| 22/09/2026 | 02:00 | 295 | 2,68 | 16 | 12,53 |
| 22/09/2026 | 03:00 | 293 | 3,22 | 15 | 14,84 |
| 22/09/2026 | 04:00 | 295 | 4,60 | 21 | 20,03 |
| 22/09/2026 | 05:00 | 293 | 10,94 | 44 | 39,91 |
| 22/09/2026 | 06:00 | 295 | 30,01 | 58 | 85,36 |
| 22/09/2026 | 07:00 | 293 | 36,65 | 54 | 99,57 |
| 22/09/2026 | 08:00 | 295 | 39,00 | 56 | 111,46 |
| 22/09/2026 | 09:00 | 295 | 37,62 | 59 | 118,25 |
| 22/09/2026 | 10:00 | 281 | 35,27 | 57 | 114,90 |
| 22/09/2026 | 11:00 | 294 | 36,77 | 55 | 132,10 |
| 22/09/2026 | 12:00 | 296 | 33,80 | 53 | 122,60 |
| 22/09/2026 | 13:00 | 295 | 34,26 | 52 | 110,27 |
| 22/09/2026 | 14:00 | 295 | 35,80 | 51 | 115,30 |
| 22/09/2026 | 15:00 | 295 | 36,57 | 54 | 116,50 |
| 22/09/2026 | 16:00 | 285 | 32,75 | 55 | 115,65 |
| 22/09/2026 | 17:00* | 204 | 25,64 | 50 | 95,24 |

*Hai khung 17:00 chỉ có dữ liệu một phần giờ: ngày 21/09 bắt đầu từ 17:41:40; ngày 22/09 kết thúc lúc 17:41:32. Không nên so sánh chúng như hai giờ quan sát đầy đủ.*

Số xe giảm dần từ tối đến rạng sáng. Trung bình tăng từ 10,94 xe ở 05:00–06:00 lên 30,01 xe ở 06:00–07:00, sau đó đạt 39,00 xe ở 08:00–09:00. Từ 09:00 đến trước 16:00, mức trung bình vẫn khoảng 34–38 xe/khung hình. Đây là diễn biến của ngày quan sát, chưa phải bằng chứng về quy luật lặp lại hằng ngày.

## 7. Thành phần phương tiện

| Loại phương tiện | Tỷ trọng lượt phát hiện |
|---|---:|
| Ô tô | 52,60% |
| Xe máy | 32,73% |
| Xe tải | 10,19% |
| Xe buýt | 4,47% |

Tổng tỷ lệ sau làm tròn có thể lệch nhẹ so với 100%. Không tính `person` vào mẫu số.

Xe máy tăng từ trung bình 3,35 xe/ảnh ở 05:00–06:00 lên 14,50 ở 06:00–07:00 và 17,79 ở 07:00–08:00. Xe tải có trung bình cao nhất ở 11:00–12:00, đạt 6,51 xe/ảnh.

Tỷ trọng này mô tả kết quả phát hiện, chưa phải cơ cấu lưu lượng thực tế. Sự che khuất, kích thước đối tượng và chất lượng ảnh có thể làm độ chính xác khác nhau giữa các loại xe. CSV không có nhãn đối chiếu để định lượng những sai số đó.

## 8. Phân tích WTI và WTI_norm

Công thức đang dùng:

```text
total = car + motorbike + bus + truck
WTI = 4 × car + motorbike + 7 × bus + 6 × truck
WTI_norm = WTI × (1 + ln(1 + total))
```

`ln` là logarit tự nhiên. Kiểm tra lại cho thấy cả 7.044 dòng khớp công thức; so sánh WTI_norm dùng dung sai số thực qua `numpy.isclose`.

WTI đặt trọng số lớn hơn cho ô tô, xe buýt và xe tải so với xe máy. Vì thế hai ảnh có cùng tổng số xe có thể có WTI khác nhau đáng kể. Dữ liệu không xác nhận rằng các trọng số 4, 1, 7, 6 đã được hiệu chuẩn theo diện tích chiếm dụng hoặc mức ùn tắc.

WTI_norm không bị giới hạn trong một thang cố định. Với total lớn hơn 0, hệ số nhân lớn hơn 1, nên đây là phép khuếch đại phi tuyến, không phải chuẩn hóa min–max hay z-score.

Khung 08:00–09:00 có total trung bình cao nhất (39,00), nhưng khung 11:00–12:00 có WTI trung bình cao nhất (132,10). Thành phần phương tiện, đặc biệt số xe tải cao vào cuối buổi sáng, góp phần giải thích sự khác biệt.

## 9. Tương quan và các mẫu cực đại

Các tương quan Pearson đáng chú ý:

| Cặp biến | Hệ số tương quan |
|---|---:|
| WTI và WTI_norm | 0,996 |
| total và WTI_norm | 0,944 |
| total và WTI | 0,929 |
| car và WTI | 0,902 |
| person và motorbike | 0,893 |
| car và total | 0,835 |
| person và total | 0,811 |
| motorbike và total | 0,802 |

WTI, WTI_norm và total được tạo từ các cột đếm nên tương quan cao một phần đến từ quan hệ công thức. Không thể dùng tương quan này làm bằng chứng rằng chỉ số dự báo tốt ùn tắc thực tế. Person và motorbike cùng biến động mạnh, nhưng CSV không đủ để xác định nguyên nhân hay suy ra quan hệ nhân quả. Các hệ số cũng có thể chịu ảnh hưởng của sự khác biệt ngày–đêm.

Năm mẫu có total cao nhất:

| Timestamp | Car | Motorbike | Bus | Truck | Total | WTI |
|---|---:|---:|---:|---:|---:|---:|
| 22/09/2026 09:20:57 | 21 | 37 | 0 | 1 | 59 | 127 |
| 22/09/2026 06:56:54 | 17 | 39 | 1 | 1 | 58 | 120 |
| 22/09/2026 10:25:00 | 25 | 22 | 3 | 7 | 57 | 185 |
| 22/09/2026 08:51:35 | 17 | 36 | 1 | 2 | 56 | 123 |
| 22/09/2026 08:56:29 | 24 | 28 | 0 | 4 | 56 | 148 |

WTI và WTI_norm cùng đạt cực đại lúc 11:50:29 ngày 22/09/2026: 24 ô tô, 1 xe máy, 8 xe buýt và 19 xe tải, tổng 52 xe; WTI = 267 và WTI_norm ≈ 1.327,07. Đây là mẫu nên đối chiếu ảnh do số xe hạng nặng cao, chưa thể coi là lỗi chỉ từ giá trị lớn.

## 10. Phân vị và ngưỡng đông xe tham khảo

| Phân vị | Total | WTI | WTI_norm |
|---|---:|---:|---:|
| P05 | 1 | 4 | 6,77 |
| P50 | 23 | 77 | 321,54 |
| P85 | 40 | 129 | 602,30 |
| P90 | 42 | 139 | 652,60 |
| P95 | 46 | 154 | 732,18 |
| P99 | 51 | 184 | 888,59 |

Nếu áp dụng điều kiện vượt ngưỡng nghiêm ngặt:

| Điều kiện | Số mẫu | Tỷ lệ mẫu |
|---|---:|---:|
| total > 40, tức từ 41 xe | 993 | 14,10% |
| total > 42, tức từ 43 xe | 682 | 9,68% |

Tỷ lệ không nhất thiết đúng 15% hay 10% vì total là số nguyên và nhiều mẫu bằng ngưỡng. Đây là tỷ lệ bản ghi, không phải tỷ lệ thời gian chính xác.

Các ngưỡng được tính trên toàn bộ khoảng quan sát, bao gồm ban đêm ít xe. Chúng không phải phân vị lịch sử theo từng khung giờ như định hướng trong README. Một ngưỡng chung có thể phản ánh chênh lệch ngày–đêm thay vì sự bất thường trong một khung giờ cụ thể.

Số xe lớn chưa đủ để kết luận ùn tắc: CSV không có tốc độ, thời gian chờ hoặc nhãn ùn tắc đã đối chiếu. Vì vậy nên gọi đây là ngưỡng đông xe thử nghiệm cho đến khi có kiểm chứng.

## 11. Hàm ý cho bài toán dự báo thời gian tới ùn tắc

### Dữ liệu cần bổ sung

- Thu thập nhiều ngày, bao gồm các ngày trong tuần và cuối tuần, để đánh giá mức độ lặp lại và xây dựng đường cơ sở theo giờ.
- Lưu định danh camera, thời điểm ảnh nguồn nếu có, thời điểm xử lý, phiên bản mô hình và cấu hình nhận diện để truy vết thay đổi.
- Lưu thông tin chất lượng ảnh và trạng thái thu thập để phân biệt khoảng thiếu dữ liệu với đường vắng.
- Đối chiếu ảnh đại diện cho ban ngày, ban đêm, mẫu 0 xe và các đỉnh cao với số đếm thủ công.

### Chuẩn bị chuỗi thời gian

- Tổng hợp theo phút bằng trung bình hoặc trung vị số xe/ảnh để giảm dao động; không diễn giải tổng các số đếm ảnh trong phút thành lưu lượng xe duy nhất.
- Ghi kèm số mẫu mỗi phút để nhận biết độ phủ và giữ dấu các khoảng thiếu. Không tự điền 0 vào phút không có dữ liệu.
- Tạo đặc trưng trễ và rolling theo thời gian thực, chẳng hạn 5/15/30 phút, phù hợp với nhịp lấy mẫu không hoàn toàn đều.
- Chỉ sử dụng quan sát đã có tại thời điểm dự báo. Tránh rolling có tâm hoặc các phép biến đổi sử dụng tương lai.

### Định nghĩa nhãn và đánh giá

- Chọn rõ chỉ số, điều kiện vượt ngưỡng và yêu cầu duy trì vượt ngưỡng trong bao lâu. Một ảnh tăng đột ngột không nhất thiết là một đợt ùn tắc.
- Khi tạo nhãn thời gian tới sự kiện, không nối xuyên khoảng mất dữ liệu nếu không biết điều gì xảy ra trong khoảng đó.
- Với các mẫu cuối chuỗi chưa quan sát được sự kiện tiếp theo, đánh dấu nhãn chưa xác định hoặc xử lý kiểm duyệt phù hợp; không mặc định bằng 0.
- Chia train/validation/test theo thời gian. Tính ngưỡng, đường cơ sở và tham số chuẩn hóa từ tập huấn luyện để tránh rò rỉ thông tin từ tập đánh giá.
- WTI và WTI_norm tương quan rất cao, nên đánh giá giá trị bổ sung của từng biến thay vì mặc định cần cả hai.
- So sánh mô hình với dự báo cơ sở đơn giản trên dữ liệu tương lai. Chưa có mô hình nào được huấn luyện hoặc đánh giá trong báo cáo này.

## 12. Giới hạn và phạm vi kết luận

Báo cáo xác nhận cấu trúc dữ liệu nhất quán, nhịp quan sát chủ yếu khoảng 12 giây và sự thay đổi số xe rõ giữa các thời điểm trong gần một ngày. Báo cáo chưa kiểm chứng độ chính xác YOLO, chưa xác nhận ùn tắc từ ảnh hoặc tốc độ, chưa kiểm định tính dừng hay chu kỳ, và chưa đánh giá khả năng dự báo.

Toàn bộ kết quả là thống kê mô tả của phiên bản `runs/data.csv` đã đọc. File dữ liệu nguồn và mã chương trình không được chỉnh sửa trong quá trình lập báo cáo.

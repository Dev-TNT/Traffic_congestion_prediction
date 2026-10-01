# Phân tích dữ liệu giao thông ngày thứ hai

Ngày phân tích: 30/09/2026. Nguồn: `data/data2.csv` (7.065 bản ghi). Công thức chỉ số được đối chiếu với `core/Yolo_detect.py`; quy trình ghi dữ liệu được tham khảo từ `get_data.py`.

## 1. Kết quả chính

- Dữ liệu bao phủ gần 24 giờ, từ 22:39:52 ngày 28/09/2026 đến 22:39:43 ngày 29/09/2026.
- Số phương tiện trung bình là 22,95 xe/khung hình; trung vị 24 xe. Số xe thấp nhất vào khoảng 02:00–03:00 và tăng rõ từ 05:00.
- Trong các giờ đầy đủ, số xe trung bình cao nhất là 16:00–17:00 ngày 29/09, đạt 38,41 xe/khung hình; thấp nhất là 02:00–03:00, đạt 4,34 xe/khung hình.
- Đỉnh tức thời là 59 xe lúc 17:22:01 ngày 29/09/2026. WTI cao nhất là 217 lúc 09:33:53.
- Không phát hiện ô thiếu, dòng trùng hoàn toàn, timestamp trùng, số đếm âm hoặc sai lệch công thức `total`, `WTI`, `WTI_norm`.
- `WTI_norm` là phép khuếch đại WTI theo tổng số xe, không phải chuẩn hóa về 0–1 hay 0–100.
- Một ngày quan sát chỉ phù hợp với phân tích mô tả và thử nghiệm ban đầu; chưa đủ để xác nhận quy luật theo ngày trong tuần, ngưỡng ùn tắc hay chất lượng dự báo.

## 2. Phạm vi và cách phân tích

Phân tích đọc toàn bộ CSV, chuyển `timestamp` thành thời gian và kiểm tra kiểu dữ liệu, giá trị thiếu, trùng lặp, miền giá trị, thứ tự thời gian và khoảng cách giữa các mẫu. Các cột dẫn xuất được tính lại từ số đếm phương tiện để đối chiếu.

Thống kê theo giờ được nhóm theo ngày và giờ thực tế. Trung bình, phân vị và tương quan được tính theo bản ghi; trung bình theo giờ không được gia quyền theo thời lượng giữa các mẫu. Tỷ trọng phương tiện là tổng số lượt phát hiện của từng loại chia cho tổng `total` trên toàn bộ dữ liệu.

Các giá trị trình bày được làm tròn; kiểm tra công thức dùng dữ liệu chưa làm tròn và `numpy.isclose` cho số thực. Tương quan sử dụng hệ số Pearson; phân vị sử dụng nội suy tuyến tính mặc định của pandas.

Timestamp không chứa múi giờ nên báo cáo giữ nguyên thời gian ghi trong file. CSV không có `camera_id`, nhãn ùn tắc, tốc độ, ảnh nguồn hoặc mã phiên thu thập để tự xác minh bối cảnh từng mẫu.

## 3. Cấu trúc dữ liệu

| Cột | Ý nghĩa |
|---|---|
| `timestamp` | Thời điểm ghi nhận, định dạng `YYYY-MM-DD HH:MM:SS` |
| `person` | Số đối tượng được nhận diện là người |
| `car` | Số ô tô được nhận diện |
| `motorbike` | Số xe máy được nhận diện |
| `bus` | Số xe buýt được nhận diện |
| `truck` | Số xe tải được nhận diện |
| `total` | Tổng `car + motorbike + bus + truck`, không gồm `person` |
| `WTI` | Tổng số phương tiện có trọng số theo loại xe |
| `WTI_norm` | WTI nhân với hệ số phụ thuộc logarit của tổng số xe |

Các số đếm là số đối tượng trong từng ảnh. Cùng một phương tiện có thể xuất hiện trong nhiều ảnh, vì vậy tổng qua nhiều dòng không phải số phương tiện duy nhất hay lưu lượng xe đi qua giao lộ.

## 4. Chất lượng dữ liệu và độ liên tục

| Kiểm tra | Kết quả |
|---|---:|
| Số dòng dữ liệu | 7.065 |
| Số cột | 9 |
| Ô thiếu | 0 |
| Timestamp không phân tích được | 0 |
| Timestamp trùng | 0 |
| Dòng trùng hoàn toàn | 0 |
| Giá trị âm trong các cột số | 0 |
| Số đếm không nguyên | 0 |
| Dòng sai công thức `total` | 0 |
| Dòng sai công thức `WTI` | 0 |
| Dòng sai công thức `WTI_norm` trong dung sai số thực | 0 |

Timestamp tăng đúng thứ tự. Thời gian bao phủ là 23 giờ 59 phút 51 giây, gồm 7.064 khoảng cách giữa hai bản ghi liên tiếp.

| Khoảng cách giữa hai mẫu | Giây |
|---|---:|
| Trung bình | 12,23 |
| Độ lệch chuẩn | 3,24 |
| Nhỏ nhất | 3 |
| Phân vị 25% | 12 |
| Trung vị | 12 |
| Phân vị 75% | 13 |
| Lớn nhất | 170 |

Ba khoảng gián đoạn dài hơn 60 giây:

| Mẫu trước | Mẫu sau | Khoảng cách |
|---|---|---:|
| 29/09/2026 10:00:26 | 29/09/2026 10:02:52 | 146 giây |
| 29/09/2026 16:04:09 | 29/09/2026 16:06:35 | 146 giây |
| 29/09/2026 20:00:05 | 29/09/2026 20:02:55 | 170 giây |

CSV không đủ thông tin để xác định nguyên nhân gián đoạn. Quy trình thu thập chỉ ghi sau khi xử lý ảnh mới, nên không được diễn giải khoảng trống là đường không có xe.

Có 86 mẫu `total = 0`, tương đương 1,22%. Cần đối chiếu ảnh để phân biệt đường vắng, lỗi nhận diện và ảnh không sử dụng được. Có 7 cặp mẫu liên tiếp giống nhau ở cả năm cột đếm; số đếm giống nhau không chứng minh ảnh trùng và không đủ lý do để xóa mẫu. Tính nhất quán số học cũng không chứng minh YOLO đếm chính xác.

## 5. Thống kê mô tả

Đơn vị các cột đếm là đối tượng/khung hình. WTI và WTI_norm là chỉ số theo công thức dự án, chưa phải đại lượng giao thông đã hiệu chuẩn.

| Biến | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P25 | Trung vị | P75 | Lớn nhất |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 8,43 | 6,10 | 0 | 3 | 8 | 13 | 28 |
| car | 11,38 | 7,96 | 0 | 4 | 11 | 17 | 38 |
| motorbike | 8,47 | 7,11 | 0 | 2 | 7 | 13 | 37 |
| bus | 0,89 | 1,35 | 0 | 0 | 0 | 1 | 12 |
| truck | 2,21 | 2,37 | 0 | 0 | 2 | 3 | 17 |
| total | 22,95 | 13,54 | 0 | 11 | 24 | 34 | 59 |
| WTI | 73,49 | 44,37 | 0 | 34 | 74 | 109 | 217 |
| WTI_norm | 320,40 | 219,78 | 0 | 118,30 | 310,38 | 496,53 | 1.086,59 |

Khoảng 50% số mẫu có 11–34 xe. Dao động lớn phản ánh cả khác biệt theo thời điểm và biến động giữa các ảnh; riêng thống kê này không xác định được tác động của đèn tín hiệu, chuyển động giao thông hay sai số nhận diện.

## 6. Diễn biến theo giờ

| Ngày | Giờ bắt đầu | Số mẫu | Total trung bình | Total lớn nhất | WTI trung bình |
|---|---|---:|---:|---:|---:|
| 28/09/2026 | 22:00* | 101 | 13,84 | 33 | 44,50 |
| 28/09/2026 | 23:00 | 294 | 10,63 | 27 | 34,33 |
| 29/09/2026 | 00:00 | 295 | 8,08 | 25 | 26,61 |
| 29/09/2026 | 01:00 | 297 | 5,30 | 19 | 19,17 |
| 29/09/2026 | 02:00 | 297 | 4,34 | 14 | 16,60 |
| 29/09/2026 | 03:00 | 296 | 4,86 | 14 | 18,46 |
| 29/09/2026 | 04:00 | 296 | 7,53 | 20 | 27,67 |
| 29/09/2026 | 05:00 | 297 | 17,13 | 45 | 53,05 |
| 29/09/2026 | 06:00 | 296 | 26,88 | 53 | 79,87 |
| 29/09/2026 | 07:00 | 295 | 30,18 | 47 | 99,74 |
| 29/09/2026 | 08:00 | 294 | 32,20 | 51 | 106,71 |
| 29/09/2026 | 09:00 | 295 | 34,17 | 57 | 116,01 |
| 29/09/2026 | 10:00 | 286 | 32,82 | 53 | 110,27 |
| 29/09/2026 | 11:00 | 295 | 32,80 | 54 | 109,98 |
| 29/09/2026 | 12:00 | 296 | 33,46 | 51 | 111,96 |
| 29/09/2026 | 13:00 | 296 | 31,51 | 52 | 101,54 |
| 29/09/2026 | 14:00 | 296 | 32,52 | 50 | 107,58 |
| 29/09/2026 | 15:00 | 296 | 33,26 | 53 | 108,23 |
| 29/09/2026 | 16:00 | 283 | 38,41 | 55 | 121,08 |
| 29/09/2026 | 17:00 | 297 | 34,10 | 59 | 97,55 |
| 29/09/2026 | 18:00 | 296 | 26,09 | 52 | 80,98 |
| 29/09/2026 | 19:00 | 295 | 24,16 | 45 | 69,73 |
| 29/09/2026 | 20:00 | 283 | 19,28 | 44 | 53,69 |
| 29/09/2026 | 21:00 | 296 | 18,42 | 36 | 50,59 |
| 29/09/2026 | 22:00* | 197 | 13,69 | 35 | 45,93 |

*Hai khung 22:00 chỉ có dữ liệu một phần giờ: ngày 28/09 bắt đầu lúc 22:39:52; ngày 29/09 kết thúc lúc 22:39:43. Không nên so sánh chúng như hai giờ đầy đủ.*

Số xe giảm dần sau 23:00 và chạm mức thấp nhất lúc 02:00–03:00. Trung bình tăng từ 7,53 xe lúc 04:00–05:00 lên 17,13 lúc 05:00–06:00 và 26,88 lúc 06:00–07:00. Ban ngày duy trì khoảng 30–34 xe/khung hình trước khi đạt 38,41 lúc 16:00–17:00, rồi giảm dần vào buổi tối. Đây chỉ là diễn biến của ngày quan sát.

## 7. Thành phần phương tiện

| Loại phương tiện | Tỷ trọng lượt phát hiện |
|---|---:|
| Ô tô | 49,61% |
| Xe máy | 36,89% |
| Xe tải | 9,64% |
| Xe buýt | 3,86% |

Không tính `person` vào mẫu số. Tổng tỷ lệ sau làm tròn có thể lệch nhẹ so với 100%.

Xe máy có trung bình theo giờ cao nhất lúc 17:00–18:00, đạt 15,66 xe/ảnh. Xe tải cao nhất lúc 12:00–13:00, đạt 4,92 xe/ảnh. Tỷ trọng trên mô tả kết quả phát hiện chứ chưa phải cơ cấu lưu lượng thực tế; CSV không có nhãn thủ công để đo độ chính xác theo từng loại xe.

## 8. Phân tích WTI và WTI_norm

Công thức đang dùng:

```text
total = car + motorbike + bus + truck
WTI = 4 × car + motorbike + 7 × bus + 6 × truck
WTI_norm = WTI × (1 + ln(1 + total))
```

`ln` là logarit tự nhiên. Cả 7.065 dòng khớp ba công thức. WTI gán trọng số lớn hơn cho ô tô, xe buýt và xe tải nên hai ảnh có cùng `total` có thể có WTI khác nhau. Dữ liệu chưa chứng minh các trọng số đã được hiệu chuẩn theo diện tích chiếm dụng hoặc mức ùn tắc.

WTI_norm không bị giới hạn trong thang cố định. Khi `total > 0`, hệ số nhân lớn hơn 1; đây là phép khuếch đại phi tuyến chứ không phải min–max hay z-score.

Khung 16:00–17:00 vừa có `total` trung bình cao nhất (38,41), vừa có WTI trung bình cao nhất (121,08). Tuy nhiên, mẫu `total` lớn nhất và mẫu WTI lớn nhất không cùng thời điểm, cho thấy thành phần phương tiện ảnh hưởng rõ tới WTI.

## 9. Tương quan và các mẫu cực đại

Các tương quan Pearson đáng chú ý:

| Cặp biến | Hệ số tương quan |
|---|---:|
| WTI và WTI_norm | 0,997 |
| total và WTI_norm | 0,943 |
| total và WTI | 0,928 |
| car và WTI | 0,907 |
| car và total | 0,838 |
| person và motorbike | 0,827 |
| person và total | 0,775 |
| motorbike và total | 0,765 |

WTI, WTI_norm và total được tạo từ các cột đếm nên tương quan cao phần lớn do quan hệ công thức. Không thể dùng các hệ số này làm bằng chứng dự báo tốt hoặc quan hệ nhân quả.

Năm mẫu có `total` cao nhất:

| Timestamp | Car | Motorbike | Bus | Truck | Total | WTI |
|---|---:|---:|---:|---:|---:|---:|
| 29/09/2026 17:22:01 | 19 | 33 | 4 | 3 | 59 | 155 |
| 29/09/2026 09:44:15 | 20 | 26 | 0 | 11 | 57 | 172 |
| 29/09/2026 17:36:37 | 27 | 21 | 8 | 0 | 56 | 185 |
| 29/09/2026 16:23:35 | 20 | 22 | 3 | 10 | 55 | 183 |
| 29/09/2026 17:12:18 | 25 | 26 | 1 | 3 | 55 | 151 |

WTI và WTI_norm cùng đạt cực đại lúc 09:33:53 ngày 29/09/2026: 32 ô tô, 9 xe máy, 2 xe buýt và 11 xe tải, tổng 54 xe; WTI = 217 và WTI_norm ≈ 1.086,59. Nên đối chiếu ảnh nguồn do số xe hạng nặng và ô tô cao, nhưng không thể kết luận đây là lỗi chỉ từ giá trị lớn.

## 10. Phân vị và ngưỡng đông xe tham khảo

| Phân vị | Total | WTI | WTI_norm |
|---|---:|---:|---:|
| P05 | 2 | 8 | 16,79 |
| P50 | 24 | 74 | 310,38 |
| P85 | 38 | 124 | 571,75 |
| P90 | 40 | 132 | 618,92 |
| P95 | 43 | 144 | 680,87 |
| P99 | 49 | 167 | 805,23 |

Nếu áp dụng điều kiện vượt ngưỡng nghiêm ngặt:

| Điều kiện | Số mẫu | Tỷ lệ mẫu |
|---|---:|---:|
| total > 38, tức từ 39 xe | 981 | 13,89% |
| total > 40, tức từ 41 xe | 667 | 9,44% |

Tỷ lệ không nhất thiết đúng 15% hoặc 10% vì `total` là số nguyên và nhiều mẫu bằng ngưỡng. Đây là tỷ lệ bản ghi, không phải tỷ lệ thời gian chính xác.

Các ngưỡng được tính trên toàn bộ ngày, gồm cả ban đêm ít xe. Chúng chỉ là ngưỡng đông xe tham khảo, không phải ngưỡng ùn tắc đã xác minh. Muốn kết luận ùn tắc cần thêm tốc độ, thời gian chờ, mức chiếm dụng hoặc nhãn đối chiếu.

## 11. Hàm ý cho bài toán dự báo thời gian tới ùn tắc

### Dữ liệu cần bổ sung

- Thu thập nhiều ngày, gồm ngày thường và cuối tuần, để kiểm tra tính lặp lại và xây dựng đường cơ sở theo giờ.
- Lưu `camera_id`, thời điểm ảnh nguồn, thời điểm xử lý, phiên bản mô hình và cấu hình nhận diện.
- Lưu trạng thái thu thập và chất lượng ảnh để phân biệt mất dữ liệu với đường vắng.
- Đối chiếu thủ công ảnh ban ngày, ban đêm, mẫu 0 xe và các cực đại; bổ sung nhãn ùn tắc hoặc đại lượng giao thông độc lập.

### Chuẩn bị chuỗi thời gian

- Tổng hợp theo phút bằng trung bình hoặc trung vị số xe/ảnh để giảm dao động; không cộng số đếm ảnh thành số xe duy nhất.
- Ghi số mẫu mỗi phút, giữ dấu khoảng thiếu và không tự điền 0 khi không có quan sát.
- Tạo đặc trưng trễ và rolling theo thời gian thực, chẳng hạn 5/15/30 phút; chỉ dùng dữ liệu đã có tại thời điểm dự báo.
- Không dùng `WTI_norm` như một nguồn thông tin độc lập nếu đã có WTI và total vì nó được tính hoàn toàn từ hai biến này.

### Định nghĩa nhãn và đánh giá

- Xác định rõ tiêu chí sự kiện, ngưỡng và thời gian duy trì; một ảnh tăng đột ngột chưa chắc là ùn tắc.
- Không nối nhãn xuyên qua khoảng mất dữ liệu nếu không biết trạng thái trong khoảng đó.
- Với cuối chuỗi chưa quan sát đủ tương lai, đánh dấu nhãn chưa xác định thay vì mặc định không có sự kiện.
- Chia train/validation/test theo thời gian; chỉ tính ngưỡng và tham số chuẩn hóa từ tập huấn luyện để tránh rò rỉ.
- So sánh với baseline đơn giản và đánh giá trên nhiều ngày tương lai. Báo cáo này chưa huấn luyện mô hình.

## 12. Giới hạn và phạm vi kết luận

Báo cáo xác nhận cấu trúc dữ liệu nhất quán, nhịp quan sát chủ yếu 12–13 giây và biến động số xe rõ theo thời điểm trong gần một ngày. Báo cáo chưa kiểm chứng độ chính xác YOLO, chưa xác nhận ùn tắc từ ảnh hoặc tốc độ, chưa kiểm định chu kỳ và chưa đánh giá khả năng dự báo.

Toàn bộ kết quả là thống kê mô tả của phiên bản `data/data2.csv` đã đọc. File dữ liệu nguồn và mã chương trình không được chỉnh sửa trong quá trình lập báo cáo.

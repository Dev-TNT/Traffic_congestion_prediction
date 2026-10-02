# Rà soát dự báo WTI và cảnh báo giao thông — 2026-10-02

## Phạm vi và kết luận

Đọc pipeline lấy ảnh → YOLO → WTI → tổng hợp phút → dự báo → ETA → UI, script training và metric hiện có. Load model và chạy inference lại trên toàn bộ 956 cửa sổ test hợp lệ; không train, không đổi code vận hành hoặc môi trường.

Hệ thống hiện dự báo WTI, một điểm đếm phương tiện có trọng số, chưa phải dự báo kẹt xe có nhãn thực tế. Có ba vấn đề riêng biệt: dự báo tạm thời dùng quá ít thông tin realtime; model chưa hơn baseline ở mục tiêu cảnh báo sớm; dữ liệu đầu vào chưa đo được trạng thái giao thông cần dự báo.

Ảnh người dùng gửi ghi `TẠM THỜI`, còn 56/61 phút tham chiếu. Đường cam trong ảnh là đường tham chiếu nhân hệ số, không phải đầu ra sklearn. Chưa có 60 phút thực tế tương lai tương ứng để tính sai số của riêng ảnh này.

## Model và môi trường thực tế

- Model: `models/traffic_wti_forecast.joblib`, metadata `traffic-wti-v1`.
- `MultiOutputRegressor(HistGradientBoostingRegressor(...))`, 60 bộ hồi quy trực tiếp, 61 feature, target WTI_mean; validation chọn chế độ absolute thay vì delta.
- Không dùng WTI_norm làm target. Công thức WTI = 4×car + motorbike + 7×bus + 6×truck.
- Conda `traffic-congestion-prediction`, sklearn 1.9.1: load thành công, offline inference trả 60 timestamp và 60 giá trị, không provisional.
- `.venv`, sklearn 1.7.2: xuất cảnh báo khác phiên bản rồi load thất bại ở dữ liệu pandas StringDtype. Đây là lỗi tương thích bundle/môi trường đã tái hiện; chưa cô lập riêng phần đóng góp của pandas và sklearn. Python mặc định hệ thống không có sklearn.
- Chưa xác định interpreter của cửa sổ app trong ảnh. Không quy lỗi môi trường cho ảnh chỉ từ ảnh. Cần chạy train và app cùng môi trường, lưu phiên bản Python/numpy/pandas/sklearn/joblib vào manifest, kiểm tra tương thích và hiển thị lỗi load cụ thể. sklearn không hỗ trợ load model giữa các phiên bản khác nhau: [model persistence](https://scikit-learn.org/stable/model_persistence.html).

## Kết quả kiểm chứng

SHA256 của hai CSV hiện tại khớp training report trong bundle. Test không được dùng để fit trong script hiện tại; train/test cách nhau 8.940 phút. Feature dùng quá khứ, target dùng tương lai, validation loại các origin có target chạm vào phần validation. Chưa thấy leakage tương lai rõ ràng trong các hàm này.

| Dữ liệu | Mẫu ảnh | Phút | Phút mất mẫu | Origin dùng được |
|---|---:|---:|---:|---:|
| data1 | 7.044 | 1.439 | 2 | 1.098 |
| data2 | 7.065 | 1.439 | 3 | 956 |

Khoảng cách mẫu trung vị 12 giây, p95 14 giây ở cả hai tập; khoảng mất dài nhất lần lượt 147 và 170 giây. Đa số phút có 5 mẫu, nhưng có phút chỉ 1–2 mẫu. Validation chỉ nằm trong 09:21–15:01 ngày 22/09, không đại diện đầy đủ các giờ cao điểm. Hai capture gần 24 giờ không phải nhiều ngày độc lập; hàng nghìn cửa sổ chồng lấn không tạo ra hàng nghìn tình huống giao thông độc lập.

MAE dưới đây tính bằng WTI, dự báo đã làm mượt 3 phút, đối chiếu với WTI trung bình phút gốc, không làm mượt ground truth:

| Phương pháp | +1 | +15 | +30 | +60 | MAE trung bình 60 horizon | F1 vượt ngưỡng |
|---|---:|---:|---:|---:|---:|---:|
| Model | 13,70 | 13,40 | 14,10 | 14,71 | 14,12 | 0,955 |
| Cùng giờ data1 | 14,27 | 14,77 | 14,88 | 14,35 | 14,71 | 0,971 |
| Giữ nguyên giá trị hiện tại | 15,72 | 17,04 | 17,29 | 20,70 | 18,55 | 0,683 |
| Ngoại suy slope 15 phút | 15,86 | 28,23 | 38,55 | 62,13 | 40,35 | 0,721 |

Model chỉ giảm khoảng 4% MAE so với cùng giờ data1, nhưng kém hơn ở +60 và F1 vượt ngưỡng. Không nên kết luận baseline cùng giờ sẽ luôn tốt hơn: chỉ có một capture test.

Trên 956 origin: model báo sai 7, bỏ sót 39; baseline cùng giờ báo sai 0, bỏ sót 30. Đây là số cửa sổ dự báo, không phải số đợt kẹt xe. 275/530 origin có tương lai vượt ngưỡng đã có WTI hiện tại >100. F1 tổng vì vậy không phản ánh riêng khả năng báo trước một tình trạng mới.

Kiểm tra bổ sung trên 681 origin có WTI hiện tại ≤100 (chưa phải định nghĩa hoàn chỉnh của onset): model bỏ sót 29/255 origin dương tính, ETA MAE 10,93 phút trên 226 origin cảnh báo đúng. Baseline cùng giờ bỏ sót 21/255, ETA MAE 5,43 phút trên 234 origin cảnh báo đúng. Trên cùng 487 origin được cả hai cảnh báo đúng trong toàn tập: ETA MAE model 7,10 phút, baseline 3,85 phút. Những metric ETA này loại các ca bỏ sót, không phải bảo đảm sai số cho mọi cảnh báo.

Làm mượt 3 phút giảm MAE trung bình từ 14,40 xuống 14,12 nhưng làm số bỏ sót tăng 32 → 39 và ETA MAE tăng 6,26 → 7,09 phút trên các tập matched tương ứng. Đường đẹp hơn không đồng nghĩa cảnh báo tốt hơn.

## Vì sao đường tạm thời chưa bám diễn biến

`core/traffic_forecast.py:provisional_values` chỉ lấy ba phút thật cuối cùng, tính tỷ số giữa WTI thật và tham chiếu tại các phút đó, rồi nhân toàn bộ 60 phút tham chiếu tương lai với cùng tỷ số.

Hệ quả:

1. Không học xu hướng tích lũy 5, 15, 30 phút. Có 3 hay 60 phút thật vẫn dùng cùng ba phút cuối. Các metric provisional_3/5/15/30/60 trong report hiện giống hệt nhau.
2. Một nhịp đèn đỏ, lỗi nhận diện hay biến động ba phút có thể nâng/hạ cả một giờ dự báo. Không có cơ chế cho hiệu chỉnh giảm ảnh hưởng theo horizon.
3. Đường tham chiếu là 96 mức trung bình theo ô 15 phút của data1, nên dạng bậc thang. Đây chưa phải “hành vi thường nhật” được thống kê qua nhiều ngày.
4. Trên cùng 32 origin backtest khởi động, không hiệu chỉnh có MAE 14,71; hiệu chỉnh một phút là 22,74; ba phút là 16,83. Cách hiệu chỉnh hiện tại có thể làm kết quả tệ hơn. Không so trực tiếp 32 origin này với metric model trên 956 origin.
5. Phép nối làm mượt dùng hai WTI thật cuối cùng rồi đến dự báo. Nó có thể tạo điểm lõm đầu đường cam dù dự báo thô lấy mức trung bình ba phút. Đây là hệ quả có thể giải thích từ cách tính; chưa tái dựng đúng mẫu trong ảnh để khẳng định giá trị điểm đó.
6. Model chỉ được gọi khi đủ 61 phút thật liên tục. Dự báo tạm thời không phải model học trên lịch sử tham chiếu. Chuyển nhánh tại phút 61 có thể gây đổi đường đột ngột.

## Những thiếu hụt từ camera đến cảnh báo

### Đo lường và thu thập

- YOLO đếm toàn khung hình, chưa có ROI theo hướng/làn, hiệu chỉnh phối cảnh, tốc độ, độ dài hàng chờ hay thời gian đứng yên. Nhiều xe chạy nhanh và ít xe bị chặn có thể tạo WTI không phản ánh đúng mức tắc nghẽn. Chưa đánh giá detection theo ảnh có nhãn, không thể quy mọi dao động WTI cho giao thông hoặc cho YOLO.
- Chụp snapshot thưa, median khoảng 12 giây, chưa đủ căn cứ để suy ra tracking/tốc độ từng xe đáng tin cậy. Muốn dùng tốc độ/queue động cần nguồn video và kiểm chứng tracking, phối cảnh. Nếu chỉ có snapshot, ưu tiên ROI occupancy và nhãn trạng thái theo chuỗi ảnh.
- Timestamp đang gán sau YOLO, không phải thời điểm camera chụp. Không có kiểm tra tuổi ảnh phía nguồn. So sánh bytes chỉ loại được ảnh giống hoàn toàn.
- Camera 1 là nguồn dự báo cố định, đúng yêu cầu. Tuy nhiên việc lấy ảnh/YOLO cho camera đang xem nằm cùng vòng worker, có thể kéo dài chu kỳ Camera 1. Chưa đo latency/FPS thực tế để định lượng.
- Collector và ứng dụng có cách điều tiết vòng lặp khác nhau; cần đồng nhất sampling và lưu cấu hình camera/ROI/detector vào dữ liệu.
- Realtime history chỉ trong RAM. Khởi động lại phải warm-up lại; không có nhật ký forecast-origin để đối chiếu sau 5/15/60 phút.
- `get_data.py` giữ mẫu trong RAM rồi ghi cuối đợt; lỗi thoát trước bước ghi có thể mất đợt thu. Nên ghi tăng dần hoặc theo batch ngắn có checkpoint.

### Thiếu mẫu và trạng thái UI

- Một phút mất mẫu trong lịch sử 61 phút làm dự báo không sẵn sàng cho đến khi phút đó ra khỏi cửa sổ; ngược lại phút chỉ một mẫu lại được coi là hợp lệ. Chưa có điểm chất lượng/độ phủ dữ liệu.
- Provisional vẫn `is_ready=True`, có thể hiện NORMAL xanh và “Không trong 60p”. Thông báo này mạnh hơn bằng chứng hỗ trợ. Cần trạng thái “Ước lượng tạm thời — độ tin cậy thấp” và “Chưa dự báo thấy vượt ngưỡng”, không ngụ ý đảm bảo không kẹt.
- Chưa tách đang vượt ngưỡng với sắp vượt ngưỡng; chưa có hysteresis cho bật/tắt cảnh báo. ETA là đầu cửa sổ 3/5, không nhất thiết phút đầu thật sự vượt 100.
- Ngưỡng 100 chưa có nhãn thực tế để hiệu chuẩn. WTI cũng không phải lưu lượng xe/phút hoặc mật độ xe/km. FHWA ưu tiên các chỉ số gắn với thời gian hành trình/tốc độ khi đo tắc nghẽn: [congestion measures](https://ops.fhwa.dot.gov/congestion_report/chapter2.htm).

### Huấn luyện và đánh giá

- 60 HGB độc lập tối ưu sai số điểm, không tối ưu trực tiếp cảnh báo sớm/ETA hoặc tính liên tục giữa horizon. Tăng số cây không bổ sung thông tin về sự cố, tín hiệu đèn, thời tiết hoặc dòng xe sắp tới.
- Validation chọn theo MAE trung bình 60 horizon, không theo chi phí bỏ sót/báo giả. Script vẫn lưu model kể cả không hơn baseline; chưa có điều kiện chấp nhận deployment.
- MAE, F1 và ETA hiện có là đánh giá WTI vượt ngưỡng thử nghiệm. Không có nhãn kẹt xe thực nên không thể diễn giải F1=0,955 là “độ chính xác dự báo kẹt xe 95,5%”.
- Cần gom cảnh báo thành episode, đánh giá lần khởi phát khi trước đó chưa ùn, báo giả/giờ, bỏ sót episode, phân bố lead time và ETA; báo cả coverage/abstention. Tập nonoverlap hiện chỉ 16 origin, 9 dương tính.
- Đã dùng data2 để xem kết quả nhiều lần thì không nên tiếp tục coi nó là test cuối bất khả xâm phạm khi điều chỉnh thiết kế. Cần capture mới giữ kín. Duy trì chronological split và gap; hướng dẫn chính thức có ví dụ lag features và đánh giá theo thời gian: [sklearn example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).

## Phương án đề xuất, theo thứ tự

1. **Khôi phục độ tin cậy vận hành trước.** Dùng chung môi trường train/app; hiển thị nguồn dự báo, model version, số phút thật, tuổi ảnh và độ phủ. Lưu mẫu phút và từng forecast vào SQLite/CSV để khôi phục sau restart và chấm điểm khi ground truth đến. Xác minh source Camera 1 và đo cadence khi đổi camera/bật detect.
2. **Thay thuật toán khởi động bằng baseline thích nghi có kiểm chứng.** Dùng mức chênh lệch realtime so với tham chiếu và slope gần đây; cập nhật bằng exponential smoothing hoặc mô hình trạng thái level/trend, làm giảm ảnh hưởng hiệu chỉnh ở horizon xa. Trọng số phụ thuộc số phút thật và chất lượng, không nhân cố định cả giờ theo ba phút. Chọn tham số bằng nhiều origin khởi động trong train/validation; blend sang model có điều kiện, chỉ sau khi backtest chứng minh cải thiện. Không coi đây là phương án đã chứng minh thắng.
3. **Tách hai đầu ra.** Giữ model hồi quy WTI trực tiếp để vẽ; bổ sung xác suất xảy ra sự kiện trong 5/15/30/60 phút sau khi có đủ nhãn. Có thể bắt đầu LogisticRegression hoặc HistGradientBoostingClassifier, hiệu chuẩn xác suất trên validation theo thời gian. Ban đầu nhãn WTI chỉ cho phép gọi “vượt ngưỡng WTI”; muốn gọi kẹt xe phải có nhãn kẹt xe độc lập.
4. **Giữ HGB làm ứng viên, chưa chuyển ngay sang LSTM/Transformer.** So sánh persistence, cùng giờ nhiều ngày, baseline level/trend thích nghi, HGB dự báo phần dư so với baseline trên cùng các origin. Đặc trưng thêm chỉ khi dữ liệu tồn tại: ROI occupancy, queue/speed đã xác minh, thứ trong tuần, mưa, đèn tín hiệu. Không đưa tham chiếu chứa dữ liệu ngày test vào feature.
5. **Bổ sung bất định và phân biệt horizon.** Kiểm thử quantile forecasts hoặc interval được hiệu chuẩn trên validation; kiểm tra coverage theo horizon. Khoảng 5–15 phút ưu tiên cảnh báo thực dụng; 30–60 phút trình bày xu hướng và khoảng bất định. Không suy xác suất quy tắc 3/5 trực tiếp từ các quantile biên vì còn phụ thuộc tương quan giữa các phút.
6. **Thu dữ liệu và hiệu chuẩn.** Mốc khởi đầu thực dụng là 2–4 tuần cùng Camera 1/cùng ROI, nhưng không phải bảo đảm đủ mẫu. Cần nhiều đợt đông, tắc, thông thoáng, giờ cao điểm, ban đêm, mưa và ảnh lỗi, nhãn tốc độ/queue hoặc chấm tay theo quy tắc rõ. Giữ riêng các ngày test mới và theo dõi performance drift.

## Điều kiện trước khi gọi là hệ thống cảnh báo đáng tin cậy

- Detector được chấm trên ảnh cùng camera/ROI và các điều kiện khác nhau; ground truth giao thông độc lập với công thức WTI.
- Replay đúng luồng khởi động, thiếu mẫu, restart, đứt mạng, đổi camera và thay model; không để lỗi model bị hiểu thành NORMAL chắc chắn.
- Model phải vượt baseline tốt nhất theo mục tiêu đã thống nhất, trên nhiều ngày chưa dùng để chọn phương pháp. Chốt chi phí bỏ sót và giới hạn báo giả trước khi tối ưu; không chọn ngưỡng trên test.
- Báo cả sai số đường cong, số episode, báo giả/giờ, bỏ sót, ETA trên tập chung và các ca không có ETA, khoảng tin cậy/coverage. Làm mượt hiển thị và logic cảnh báo phải được kiểm chứng riêng.
- Chạy quan sát thực tế có log forecast trước khi dữ liệu tương lai xuất hiện. Unit test/import hoặc offline inference không chứng minh độ chính xác realtime.

Trong đợt rà soát này đã kiểm tra code, dữ liệu, tương thích load ở hai môi trường, tái lập inference/metric và phân tích bổ sung theo nhóm origin. Chưa chạy đo camera/YOLO trực tiếp, chưa gán nhãn ảnh, chưa kiểm chứng một giờ tương lai của ảnh gửi, chưa xác minh UI phiên đang mở. Chỉ thêm báo cáo này; không train hoặc sửa thuật toán.

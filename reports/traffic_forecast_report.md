# Báo cáo lịch sử — model total 30 phút

Phiên bản hiện hành đã chuyển sang WTI 60 phút và chưa train theo yêu cầu người dùng.
Xem [cập nhật WTI](wti_forecast_update.md). Các metric và ảnh dưới đây thuộc model cũ.

# Dự báo đường cong mật độ phương tiện và ETA vượt ngưỡng đông xe thử nghiệm

Ngày triển khai: 01/10/2026. Đây là prototype end-to-end trên hai đợt quan sát gần
24 giờ; chưa phải hệ thống dự báo ùn tắc được hiệu chuẩn.

## 1. Kiến trúc và file

```text
Camera → core/Yolo_detect.image_processing (giữ nguyên detector)
       → CameraWorker.counts_ready (mỗi ảnh mới)
       → Dashboard.sample_received (queued Qt signal)
       → ForecastWorker trên QThread riêng
           → lịch sử tham chiếu data1 → resample phút Camera 1 hoàn tất
           → giữ 61 điểm, thay dần tham chiếu bằng realtime
           → create_features dùng chung → model load một lần → 30 horizon
           → stable_crossing → ForecastResult
       → tab Dashboard hiện có → cập nhật artist Matplotlib
```

| File tạo | Vai trò |
|---|---|
| `core/traffic_features.py` | Cấu hình duy nhất, schema CSV, tổng hợp phút, feature và target |
| `core/traffic_forecast.py` | Dataclass, kiểm tra bundle, load một lần, prediction, ETA |
| `core/forecast_worker.py` | Lưu lịch sử camera, hoàn tất phút, inference ngoài UI thread |
| `ui/traffic_forecast_chart.py` | Widget PyQt5, Figure/artist được giữ nguyên |
| `scripts/train_traffic_forecast.py` | Train, baseline, metric, lưu/load kiểm chứng, bảo vệ model có sẵn |
| `scripts/backtest_traffic_forecast.py` | Backtest tĩnh tại thời điểm chọn, PNG và JSON |
| `scripts/check_traffic_ui.py` | Kiểm tra giới hạn thời gian trong cửa sổ thật; replay hoặc live |
| `tests/test_traffic_forecast.py` | 24 kiểm thử unit và integration |
| `models/traffic_forecast.joblib` | Model cùng schema, tham số, metric và nguồn dữ liệu |
| `models/traffic_forecast.metrics.json` | Metric từng horizon và baseline đầy đủ |
| `reports/traffic_backtest.png`, `.json` | Đường cong và ETA dự báo/thực tế |
| `reports/warm_start_metrics.json` | Đánh giá khởi động chỉ dùng lịch sử tham chiếu |
| `reports/ui_replay.png`, `.json` | Ảnh và dữ liệu kiểm chứng Dashboard replay |
| `reports/ui_live.png`, `.json` | Ảnh và dữ liệu kiểm chứng camera thật |
| `reports/ui_startup.png`, `.json` | Khởi động dự báo từ tham chiếu và đường thường nhật |
| `reports/traffic_forecast_report.md` | Báo cáo này |

File sửa: `main.py` (nối signals và shutdown worker), `core/vision_worker.py`
(nguồn dự báo Camera 1 độc lập với viewer), `ui/tabs/dashboard_tab.py` (thay đồ thị giả),
`utils/app_state.py` (detect viewer mặc định tắt), `ui/tabs/camera_tab.py`
(stats hiển thị -- khi detect tắt),
`requirements.txt`, `environment.yml` (chỉ thêm pandas/sklearn/joblib), `README.md`,
`.gitignore` (bỏ qua môi trường và bytecode, giữ rule trọng số YOLO).
Không sửa detector, crawler hay dữ liệu nguồn.
Các thay đổi có sẵn trong `get_data.py` và các file dữ liệu đã xóa được giữ nguyên.

## 2. Dữ liệu và chống leakage

| | Train (`data1`) | Test (`data2`) |
|---|---:|---:|
| Raw records | 7.044 | 7.065 |
| Phút sau loại biên chưa hoàn tất | 1.439 | 1.439 |
| Phút mất mẫu | 2 | 3 |
| Cửa sổ có đủ lịch sử và 30 target | 1.167 | 1.076 |
| Khoảng thời gian phút hoàn tất | 21/09 17:43 – 22/09 17:41 | 28/09 22:41 – 29/09 22:39 |

Khoảng cách train/test là **8.940 phút**, lớn hơn horizon 30. Không nối hai CSV,
không shuffle. Không fit preprocessing/scaler/ngưỡng trên test. Không dùng test
chọn hyperparameter. `early_stopping=False` tránh random validation mặc định
của sklearn; không dùng validation nội bộ trong prototype này.

Tổng hợp dùng trung bình, không cộng số đếm các ảnh thành lưu lượng. Giữ
`car_mean`, `motorbike_mean`, `bus_mean`, `truck_mean`, `total_mean`, `WTI_mean`,
`total_std` (ddof=0), `total_max`, `samples_per_minute`. Phút một mẫu có std=0;
phút không có mẫu có mean/std/max=NaN và samples=0.

Khoảng `[10:00,10:01)` có timestamp 10:01; khi đó mẫu mới khả dụng. Dự báo từ
10:01 cho timestamp 10:02…10:31. Loại phút đầu/cuối chưa hoàn tất của CSV.
Feature chỉ dùng số liệu hiện tại/quá khứ; target là `total_t+1`…`total_t+30`.
Bỏ cả cửa sổ nếu 61 phút lịch sử hoặc 30 phút target băng qua phút mất mẫu.

61 điểm là điều kiện cần để có **lag 60**; Dashboard vẽ 60 điểm cuối. Người dùng
đã chọn dùng lịch sử tham chiếu để dự báo ngay khi khởi động: bundle v2 lưu mean
của 9 cột phút theo 96 khung giờ 15 phút, chỉ fit từ `data1`. Các điểm này được
gán vào 61 timestamp trước/đến phút mở app và đánh dấu `is_reference` (không là
feature). Khi realtime Camera 1 có phút hoàn tất, điểm đó thay tham chiếu.
Sau 61 phút có quan sát liên tục, lịch sử model hoàn toàn là realtime.

Phút thiếu **sau thời điểm mở app** giữ NaN; không dùng tham chiếu để che mất
kết nối/ảnh lặp. Inference sẽ báo chưa đủ dữ liệu. Không lấy phút chưa hoàn tất
làm quan sát. Lịch sử realtime nằm trong RAM và mất khi khởi động lại; đổi camera
viewer không thay nguồn hoặc xóa lịch sử Camera 1.

Đồ thị có mốc hiện tại luôn ở giữa, trục ±60 phút. Forecast vẫn dài 30 phút.
Xanh: Camera 1 thực tế; xám chấm: lịch sử tham chiếu; cam: dự báo; đỏ: ngưỡng/ETA.
Nút **Hiện hành vi thường nhật** bật đường tím theo profile `data1` trên cả
cửa sổ. Khoảng +30…+60 không có model forecast; chỉ có profile nếu nút được bật.
Profile từ một ngày train là đường tham chiếu thử nghiệm, chưa xác nhận quy luật
hằng ngày. Trên UI luôn ghi số phút tham chiếu trong đầu vào và nhãn mật độ hiện
tại là tham chiếu hoặc realtime. Default detect của Camera tab là tắt; bật nút
mới vẽ detection của camera đang xem, còn YOLO Camera 1 chạy nền độc lập.

## 3. Model và feature

`MultiOutputRegressor(HistGradientBoostingRegressor(...), n_jobs=1)` gồm 30
regressor độc lập, dự báo trực tiếp. `max_iter=100`, `learning_rate=0.05`,
`max_leaf_nodes=15`, `min_samples_leaf=25`, `l2_regularization=2`, seed=42.
Train và inference giới hạn native CPU thread về 1 để tránh chiếm CPU của UI.

35 feature, với thứ tự cố định trong bundle:

- 9 số liệu phút hiện tại đã liệt kê ở trên.
- 8 lag: 1, 2, 3, 5, 10, 15, 30, 60 phút.
- 4 rolling mean: 5, 15, 30, 60 phút.
- 3 rolling std, 3 min, 3 max: 5, 15, 30 phút.
- 3 slope: `(total hiện tại - total trễ N) / N`, N=5,15,30.
- `hour_sin`, `hour_cos` theo thời điểm phút hoàn tất.

`person` không thuộc total. Không dùng `WTI_norm`. Không vẽ vùng bất định vì
chưa có validation/calibration độc lập để đánh giá coverage; không dựng dải
trang trí quanh dự báo rồi gọi đó là độ tin cậy.

## 4. Metric thực tế trên test

Đơn vị MAE là xe/ảnh trung bình phút, không phải phút hay xe/phút lưu lượng.

| Model/baseline | t+1 | t+5 | t+10 | t+15 | t+30 | Mean 30 horizon |
|---|---:|---:|---:|---:|---:|---:|
| HGB direct 30 horizon | 3,45 | 3,78 | 3,82 | 4,05 | 4,28 | **4,03** |
| Persistence | 4,91 | 4,62 | 4,84 | 4,87 | 5,06 | 4,99 |
| Linear trend từ 15 phút qua | 5,06 | 5,52 | 6,88 | 8,18 | 11,46 | 8,38 |
| Cùng khung giờ 15 phút, chỉ fit train | 5,04 | 5,04 | 5,02 | 4,96 | 4,87 | 4,97 |

Metric bảng này chỉ dùng lịch sử **quan sát thật** của data2, không đánh giá
warm-start bằng lịch sử tham chiếu. Model cải thiện MAE đường cong, nhưng vẫn thiếu năng lực dự báo đỉnh vượt ngưỡng.
Baseline cùng khung giờ chỉ có một đợt ~24 giờ để tham chiếu, chưa chứng minh
chu kỳ hằng ngày. Các con số chưa được kiểm định trên nhiều ngày/camera.

Ngưỡng **40** là hằng số thử nghiệm trên minute mean; không fit từ test. Sự kiện
được tính khi có ít nhất 3 điểm **>40** trong cửa sổ đủ 5 phút. ETA là phút bắt
đầu cửa sổ đầu tiên; điểm bắt đầu có thể chưa >40. Không dùng cửa sổ cuối chưa
đủ 5 phút. `WARNING` nếu ETA≤5; `WATCH` nếu ETA>5; `NORMAL` nếu không có cửa sổ
như vậy trong 30 phút dự báo. Trạng thái NORMAL chỉ mô tả đầu ra model.

| | Precision | Recall | F1 | False alarms | Missed windows | Matched ETA MAE |
|---|---:|---:|---:|---:|---:|---:|
| Model | 0 | 0 | 0 | 0 | **33** | Không đủ matched positives |
| Persistence | 0,0333 | 0,0303 | 0,0317 | 29 | 32 | 14,00 phút (1 mẫu) |
| Linear trend | 0,0734 | 0,4848 | 0,1275 | 202 | 17 | 8,0625 phút (16 mẫu) |
| Cùng khung giờ | 0 | 0 | 0 | 40 | 33 | Không đủ matched positives |

Có 33/1.076 **forecast origins** có cửa sổ vượt ngưỡng trong 30 phút tương lai.
Đây là các cửa sổ chồng lấn, **không phải 33 sự kiện giao thông độc lập**.
Precision ghi 0 theo quy ước khi không có predicted positive; không được hiểu
là model có cảnh báo để kiểm chứng. ETA MAE chỉ tính trên true positives nên
không thể thay thế missed-events metric.

Báo cáo JSON còn có view lấy mỗi 30 origins: 36 origins, MAE mean 3,94,
nhưng không có positive nên không đánh giá được khả năng cảnh báo trên view đó.
Không thay ngưỡng hoặc tune model để làm đẹp metric trên data2.

Backtest mặc định tại **29/09 09:15** cho ETA thực tế **26 phút**, model không
báo vượt ngưỡng trong 30 phút. PNG vẽ 60 phút trước, forecast 30 phút, actual
30 phút sau, ngưỡng và mốc ETA thực tế, giúp thấy rõ các đỉnh bị bỏ lỡ.

Đánh giá riêng lúc mới mở app, với **61/61 phút đầu vào là tham chiếu train-only**,
tại cùng 1.076 test origins: MAE t+1/5/10/15/30 là **4,94 / 4,94 / 4,76 / 4,66 /
4,88**, mean 30 horizon **4,84**. Precision/recall/F1 đều 0; 1 false alarm và
33 missed windows; chưa có matched positive để đo ETA MAE. Kết quả riêng ở
`reports/warm_start_metrics.json`; script train cũng đánh giá scenario này.
Đây là khởi động giả lập tại từng thời điểm, chưa phải đánh giá luồng pha trộn
tham chiếu/realtime dài hạn. Dự báo ngay khi mở app có sai số cao hơn trường
hợp đủ lịch sử quan sát thật; UI luôn đánh dấu tham chiếu.

## 5. Cách chạy

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_traffic_forecast.py
python main.py
python scripts/backtest_traffic_forecast.py
python scripts/backtest_traffic_forecast.py --at "2026-09-29 16:20" --output reports/backtest_1620.png
python -m unittest discover -s tests -v
python -m compileall -q main.py core ui scripts tests
python scripts/check_traffic_ui.py --seconds 5
python scripts/check_traffic_ui.py --startup --reference --seconds 5
python scripts/check_traffic_ui.py --live --reference --seconds 75
```

Train lại sẽ lưu candidate nếu đường dẫn mặc định đã có model. Có thể chỉ định
`--output models/experiment.joblib`; loader/backtest chấp nhận đường dẫn đó.
`--max-iter 2` dùng cho training smoke, không thay artifact 100 iterations.
Bundle có feature list, horizon, interval, history, threshold, version, metric,
tham số train, phiên bản sklearn, thời điểm tạo, SHA256 hai nguồn CSV và daily
profile chỉ fit train. Bundle v2 đã được so với v1 trên toàn bộ 1.076 test origins:
prediction của model và model metrics giống hệt; chỉ thêm reference profile.
V1 được lưu ở `models/archive/traffic_forecast.v1.joblib` trước khi đưa v2 vào
đường dẫn chính. Train tiếp theo vẫn tạo candidate thay vì ghi đè.
Loader kiểm tra schema, thứ tự feature, số output và threshold; thiếu/hỏng model
trở thành MODEL_ERROR và UI vẫn mở. Sau khi đổi threshold cần train và nạp lại.

## 6. Kiểm thử và giới hạn xác minh

Môi trường kiểm thử: Python 3.11.15, pandas 2.3.3, sklearn 1.7.2, joblib 1.5.2,
Matplotlib 3.11.1, PyQt5 5.15.11 / Qt 5.15.2, Ultralytics 8.4.137,
torch 2.13.0+cpu. `.venv` kế thừa dependency Qt/YOLO từ môi trường Conda có sẵn.
Hai dependency file giữ nguyên encoding UTF-16 và các package cũ.

- Syntax/compile: đã chạy cho entry point và các thư mục core/ui/scripts/tests.
- 24 tests: resample, phút mất mẫu, causality bằng thay dữ liệu tương lai,
  schema/order, lag 60, target horizon, ETA và equality, quy tắc 3/5, mức cảnh báo,
  thiếu lịch sử/model, bundle/output lỗi, 30 điểm và timestamp, model load một
  lần, giữ model cũ, worker hoàn tất phút, loại dữ liệu camera khác,
  widget/Dashboard queued replay, camera-worker detection contract, profile
  train-only qua nửa đêm, default detect tắt, thay dần tham chiếu, không fill
  outage sau khởi động, viewer Camera 1/2 × detect on/off, nút so sánh và trục giữa.
- Training smoke thực sự fit HGB 30 output với 2 iterations trong thư mục tạm.
  Full train 100 iterations đã lưu và reload artifact chính, đối chiếu prediction.
- UI test tạo QWidget thật, xử lý Qt events, worker ở thread khác và draw canvas;
  kiểm tra Figure và artist không đổi qua cập nhật. Không chỉ kiểm tra import.
- Native window replay mở Dashboard thật, 60 actual + 30 forecast, các thẻ
  mật độ/ngưỡng/slope/ETA/status có nội dung. Kết quả chi tiết và screenshot
  nằm ở `reports/ui_replay.json`/`.png`.
- Native startup đã xác minh 60 điểm lịch sử tham chiếu được vẽ, 30 forecast,
  đầu vào còn 61 phút tham chiếu được ghi rõ, đường thường nhật được bật và mốc
  hiện tại ở giữa. `reports/ui_startup.json`/`.png` lưu bằng chứng.
- Bounded live camera đã nhận ảnh mạng, chạy YOLO thật và phát số đếm, đồng
  thời UI timer tiếp tục hoạt động. Chi tiết lần live cuối ở `reports/ui_live.json`.
  Phiên live 75 giây nhận 7 bộ số đếm Camera 1, 7 frame; 1 phút realtime thay
  tham chiếu, đầu vào còn 60/61 phút tham chiếu và model trả 30 điểm. Có thao tác
  đổi camera viewer/chuyển tab/bật detect trong phiên; screenshot cuối là tab
  Camera, không dùng ảnh đó để khẳng định Dashboard đang hiển thị. Native startup
  và replay đã kiểm chứng Dashboard riêng. Nguồn forecast cuối vẫn Camera 1.
- **Chưa chạy đủ 61 phút camera live liên tục** để xác nhận chuyển hoàn toàn
  từ tham chiếu sang lịch sử realtime trong thử nghiệm dài. Việc thay dần tới
  61 phút đã qua unit test; inference/UI với lịch sử thật đầy đủ qua replay.
  Chưa đo latency SLA, memory soak hay chất lượng detection; đã đánh giá cold
  start toàn tham chiếu, chưa backtest toàn bộ giai đoạn pha trộn lịch sử dài hạn.
- Kiểm tra bằng mock hoặc replay không xác nhận chất lượng YOLO, không chứng
  minh khả năng dự báo ùn tắc thực tế và không thay thế thử nghiệm dài hạn.

## 7. Các việc cần làm trước production

1. Thu thập nhiều tuần, ngày thường/cuối tuần; lưu camera_id, thời gian ảnh
   nguồn, phiên/version detector, chất lượng ảnh, trạng thái thu thập và phút thiếu.
2. Đánh giá YOLO bằng nhãn thủ công theo ngày/đêm, mưa, che khuất và loại xe.
   Bổ sung nhãn ùn tắc hoặc tốc độ/thời gian chờ; hiệu chuẩn ngưỡng từng camera.
3. Thêm validation theo thời gian với gap≥30, giữ tập test cuối hoàn toàn độc
   lập. So sánh curve MAE và khả năng dự báo đỉnh/ETA, metric theo sự kiện độc
   lập, khoảng tin cậy và nhiều ngày; chưa dùng model hiện tại cho cảnh báo quan trọng.
4. Chỉ thêm prediction intervals khi có calibration riêng và kiểm chứng coverage.
   Thử mục tiêu/đặc trưng hỗ trợ đỉnh sau khi có đủ dữ liệu; không chỉnh threshold theo test.
5. Chạy ứng dụng liên tục qua chuyển lịch sử tham chiếu sang 61 phút realtime,
   đổi camera viewer, bật/tắt detect viewer, mất mạng,
   recovery và shutdown. Đo latency/memory thực tế, lưu lịch sử bền vững nếu cần
   khởi động lại không mất warm-up, giám sát drift và quy trình duyệt candidate.

## 8. Phạm vi thay đổi

Thay đồ thị giả và thêm pipeline dự báo Camera 1, default detect viewer tắt,
lịch sử tham chiếu và nút so sánh theo yêu cầu bổ sung; không sửa thuật toán YOLO/crawler,
không nâng hàng loạt dependency, không sửa CSV, không viết lại các tab khác.
File cấu hình IDE tự thêm `.venv` exclusion được đưa về nội dung trước tác vụ.
Các xóa dữ liệu và thay đổi `get_data.py` có sẵn không thuộc triển khai này.


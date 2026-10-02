# Dự báo WTI 60 phút — cập nhật 01/10/2026

Phiên bản này dự báo WTI gốc theo phút, không dùng `WTI_norm` và không dùng ảnh
đồ thị. Chưa huấn luyện model WTI mới theo yêu cầu người dùng. Những metric của
model `total` 30 phút cũ không đại diện cho phiên bản này.

## Chạy và tự huấn luyện

Từ PowerShell tại thư mục project:

```powershell
# Chỉ xem cấu trúc, không train
.\.venv\Scripts\python.exe training_model.Py --describe

# Người dùng tự chạy train
.\.venv\Scripts\python.exe training_model.Py

# Mở lại app sau khi train xong để load model một lần
.\.venv\Scripts\python.exe main.py

# Sau khi có model, kiểm tra trên ngày test
.\.venv\Scripts\python.exe scripts/backtest_traffic_forecast.py --at "2026-09-29 18:05"

# Xem giai đoạn tạm thời với 3 phút realtime, không cần model
.\.venv\Scripts\python.exe scripts/backtest_traffic_forecast.py --at "2026-09-29 18:05" --warmup-minutes 3 --output reports/wti_warmup_backtest.png
```

Code train nằm ở `training_model.Py`, hàm `build_model()` thể hiện cấu trúc
sklearn. `scripts/train_traffic_forecast.py` chỉ chuyển tiếp tới file này.
Không có hành vi train lúc import hay chạy `--describe`.

## Kiến trúc

- Camera 1 → YOLO → `WTI = 4*car + motorbike + 7*bus + 6*truck` → trung bình phút.
- Worker giữ 121 điểm (hiện tại và 120 phút trước), chỉ dùng phút đã hoàn tất.
- Model dùng 61 phút thật để tạo lag 60; trục hiển thị là ±120 phút, mốc hiện tại ở giữa.
- Target là `WTI_mean` t+1…t+60; không recursive. 61 feature gồm các loại xe,
  total/WTI mean/std/max, samples/minute, lag và rolling cho total lẫn WTI,
  slope và sin/cos giờ. Feature/target/schema có một nguồn trong `core/traffic_features.py`.
- `MultiOutputRegressor(HistGradientBoostingRegressor)` có 60 estimator trực tiếp.
  Cấu hình: 200 iterations, 15 leaves, min leaf 25, learning rate 0.05, L2 5,
  early stopping tắt để tránh random validation nội bộ; native CPU threads giới hạn 1.
- Script thử target WTI tuyệt đối và target delta = WTI tương lai − WTI hiện tại.
  Chọn mode bằng MAE sau làm mượt trên 20% cuối của **data1**, có khoảng cách
  ít nhất 60 phút giữa target cuối của fit và origin đầu của validation.
  Sau đó refit mode đã chọn trên data1. **data2 không dùng để chọn mode**.
- Dự báo delta được cộng lại WTI hiện tại bằng cùng helper cho train/inference.
  Target vẫn là WTI trung bình phút, không train trực tiếp trên từng frame.

## Giai đoạn tạm thời và làm mượt

Trước khi có đủ 61 phút Camera 1 thật, hoặc chưa có model WTI, app vẽ đường
**TẠM THỜI** nét đứt. Đây là baseline tham chiếu hiệu chỉnh, không phải đầu ra
sklearn từ lịch sử giả. Lịch sử trước lúc mở app vẫn được hiển thị riêng.

Khi chưa có phút realtime hoàn tất, dùng WTI thường nhật từ data1. Khi đã có
realtime, lấy tối đa 3 phút thật gần nhất và tính:

```
factor = mean(WTI realtime gần nhất) / mean(WTI tham chiếu tại cùng các phút)
WTI tạm thời tương lai = WTI thường nhật tương lai * factor
```

Nếu mẫu tham chiếu tại các phút đó bằng 0, dùng giá trị WTI realtime trung bình
làm baseline hằng số. Phép hiệu chỉnh chỉ dùng số liệu đến hiện tại. Nó có thể
sai khi hành vi tương lai khác ngày tham chiếu, nên UI ghi rõ nguồn tạm thời.

Khi model WTI hợp lệ và 61 phút cuối đều là realtime thật, chuyển sang model.
Không đưa các phút tham chiếu vào feature của model. Không lấp các phút mất
mẫu sau khi app mở. Mất phút hiện tại làm đường dự báo bị xóa và báo thiếu dữ liệu.

Đường realtime hiển thị trung bình trượt 3 phút trong từng đoạn dữ liệu thật,
không trộn tham chiếu hoặc vượt qua khoảng mất mẫu. Số liệu lưu và feature vẫn
là trung bình từng phút chưa làm mượt. Đường dự báo cũng dùng trailing mean 3
điểm, nối với tối đa 2 giá trị WTI thật cuối; không dùng lịch sử tham chiếu để
làm mượt điểm đầu khi đã có realtime. Nếu mới có 1–2 phút, dùng số điểm sẵn có.

Đường cam thêm điểm tại hiện tại để nối với đường xanh; điểm này không phải
horizon bổ sung. Result luôn có đúng 60 timestamp và 60 giá trị tương lai.
ETA dùng đúng 60 điểm cam đã làm mượt, theo quy tắc ít nhất 3/5 điểm > ngưỡng.

## Model, ngưỡng và đánh giá khi người dùng train

Bundle mới: `models/traffic_wti_forecast.joblib`, version `traffic-wti-v1`.
Model total cũ và các candidate được giữ nguyên; không load nhầm sang WTI.
Bundle lưu model, mode target, feature order, horizon/history/interval, smoothing,
threshold, daily profile fit trên train, phiên bản sklearn, metric và nguồn CSV.
Nếu đường dẫn output hoặc metric đã tồn tại, train tạo candidate mới và không
đưa candidate vào app tự động. File được ghi bằng chế độ exclusive.

Ngưỡng tại `CONGESTION_THRESHOLD = 100.0` trong `core/traffic_features.py`.
Đây là **điểm WTI thử nghiệm**, không phải 100 xe hay ngưỡng kẹt xe đã hiệu chuẩn.
Ngưỡng không lấy từ data2; đổi ngưỡng rồi tự train lại để metadata thống nhất.

Script in MAE +1/+5/+10/+15/+30/+60 và mean, precision/recall/F1, false alarms,
missed windows, ETA MAE trên các cửa sổ được cả actual và forecast phát hiện.
So sánh persistence, slope 15 phút và cùng giờ từ data1. Báo riêng raw/model
3-minute; MAE dùng ground truth WTI phút gốc để thấy tác động của smoothing.
Có đánh giá giai đoạn tạm thời 0/1/3/5/15/30/60 phút thật và origins cách 60 phút.
Cửa sổ event chồng lấn không được coi là các đợt ùn tắc độc lập.

## Kiểm chứng và giới hạn

- 27 unit/integration tests dùng model giả có schema 60 outputs; không gọi fit.
- Kiểm tra compile, `--describe`, causal features/target WTI, validation gap,
  serialization/load, 60 points, calibration, 3-minute smoothing, ETA, missing
  model/outage, worker thread, chart reuse, tab và Camera 1 cố định.
- Đã mở cửa sổ Qt thật với dark theme, kiểm tra đồ thị ±2 giờ và dự báo tạm thời.
- Kiểm tra live 75 giây: nguồn camera trả read timeout, không nhận frame/mẫu.
  UI vẫn phản hồi (1184 heartbeat ticks) và xóa forecast khi phút sau startup bị
  mất mẫu. Chưa xác minh được inference với số Camera 1 thật ở lần cập nhật này.
  Replay lịch sử dùng cùng worker/queued signal nhưng không được coi là kiểm chứng camera live.
- Backtest tạm thời tại 2026-09-29 18:05 với 3 phút thật: predicted ETA không có,
  actual ETA 8 phút. Đây là ví dụ phơi bày giới hạn của baseline, không phải
  metric model WTI chưa train.
- Chưa chạy training smoke hay inference với estimator WTI được fit thật vì
  người dùng yêu cầu tự train. Chưa có metric để khẳng định model mới chính xác hơn.

Dry-run dữ liệu không fit: mỗi CSV có 1439 phút; data1 có 1098 origins, data2
có 956 origins, 61 feature và 60 target. Validation có 818 fit origins / 220
validation origins; origin cuối fit cách origin đầu validation 61 phút.

Hai ngày CSV không đủ học quy luật ngày trong tuần hoặc mùa vụ. CSV chưa có
camera_id và nhãn kẹt xe thực, WTI phụ thuộc góc camera/ROI và độ chính xác YOLO.
Horizon 60 phút khó hơn 30 phút. Cần nhiều tuần dữ liệu cùng camera, nhãn thực,
time-series validation nhiều đợt, kiểm tra drift/YOLO và hiệu chuẩn ngưỡng để
đánh giá chất lượng production. Smoothing có thể giảm đỉnh và làm cảnh báo trễ.

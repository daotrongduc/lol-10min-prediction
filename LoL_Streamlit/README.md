# LoL · Dự đoán kết quả trận đấu tại phút 10

Ứng dụng **Streamlit bằng Python**, dùng mô hình thật đã huấn luyện từ dữ liệu đi kèm. Nhập thống kê của hai đội tại phút 10 để dự đoán đội thắng cuối trận. Không cần huấn luyện lại khi mở web hoặc bấm nút dự đoán.

## Trạng thái bàn giao

- Đã chạy file thực nghiệm người dùng cung cấp; mã thực nghiệm được giữ nguyên byte, chỉ chuẩn hóa tên file để import.
- Đã tạo mô hình, kết quả thực nghiệm và 20 biểu đồ từ CSV gốc.
- Đã chạy kiểm thử suy luận và các luồng giao diện bằng **Streamlit AppTest**; xem `results/web_verification.json`.
- Dự đoán web đối chiếu trên toàn bộ 988 trận Test khớp kết quả thực nghiệm đã lưu.
- Chưa triển khai trên Streamlit Community Cloud: cần chủ tài khoản đăng nhập GitHub và Streamlit, kết nối repository rồi bấm Deploy theo hướng dẫn dưới đây. Không có URL công khai đã được tạo trong lần bàn giao này.
- Chưa kiểm tra ảnh chụp giao diện trên trình duyệt thật/laptop/điện thoại vì môi trường không có công cụ kiểm thử trình duyệt phù hợp. AppTest đã chạy các trang và thao tác widget; không thay thế kiểm tra hình thức trên trình duyệt.
- Link tham khảo Ames không truy cập được qua công cụ đọc web trong lần thực hiện này. Giao diện được thiết kế riêng theo yêu cầu, không khẳng định sao chép bố cục trang tham khảo.
- Báo cáo Word đầu vào không bị sửa. Kết quả chính của lần chạy mới khớp các giá trị làm tròn trong báo cáo.

## 1. Chạy nhanh trên Windows

Cài **Python 3.12**. Giải nén ZIP, mở thư mục `LoL_Streamlit` (nơi có `app.py`), bấm vào thanh địa chỉ File Explorer, gõ `cmd`, Enter. Chạy lần lượt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Dùng các lệnh trên không cần kích hoạt môi trường nên tránh lỗi Execution Policy của PowerShell. Trình duyệt thường tự mở; nếu không, mở địa chỉ **Local URL** được Terminal in ra, thường là `http://localhost:8501`.

Trên macOS/Linux, tại thư mục dự án:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Giữ Terminal đang chạy để dùng web trên máy. Nhấn Ctrl+C để dừng. Đây là địa chỉ local, chưa phải địa chỉ `.streamlit.app` truy cập qua Internet.

**Không chỉ chạy `python app.py`**: Streamlit cần lệnh `python -m streamlit run app.py`.

## 2. Cấu trúc và vai trò các file

| Tệp/thư mục | Vai trò |
|---|---|
| `app.py` | Hai chức năng dự đoán, giao diện kết quả, kiểm tra đầu vào và cache |
| `league_of_legends_10min_data_mining.py` | Mã thực nghiệm gốc: huấn luyện/đánh giá, không tự chạy khi import |
| `results/final_model.joblib` | Dictionary chứa `model`, `feature_set`, `feature_names`, `threshold` |
| `results/` | CSV/JSON nguồn của các chỉ số hiển thị, nhật ký và cấu hình thực nghiệm |
| `figures/` | 20 hình thực nghiệm PNG |
| `samples/test_matches.csv` | Toàn bộ 988 trận Test, có nhãn để đối chiếu sau dự đoán |
| `samples/input_template.csv` | 5 trận Test mẫu, 25 biến bắt buộc và gameId; không chứa nhãn |
| `data/high_diamond_ranked_10min.csv` | CSV gốc giữ nguyên từ ZIP được cung cấp |
| `.streamlit/config.toml` | Theme sáng, màu Blue/Red, giới hạn upload 10 MB |
| `requirements.txt` | Phiên bản thư viện đã khóa |
| `tests/test_app.py` | Kiểm tra mô hình và tương tác bằng Streamlit AppTest |
| `results/input_ranges_train.csv` | Min/max của 25 biến trên Train; chỉ dùng cảnh báo, không chặn theo min/max quan sát |
| `results/web_verification.json` | Kết quả kiểm thử web và giới hạn kiểm chứng |
| `results/delivery_audit.json` | Đối chiếu nguồn, báo cáo và trạng thái bàn giao |
| `SHA256SUMS.json` | Mã băm các tệp trong gói để đối chiếu tính toàn vẹn |

Hai file Python chính là file thực nghiệm và `app.py`. File trong `tests/` phục vụ kiểm thử, không phải một mô hình hoặc ứng dụng thứ ba. Mô hình joblib không phải mã nguồn; phải giữ lại cùng dự án.

## 3. Cách dùng và demo trước giảng viên

### Dự đoán một trận

1. Trên trang đầu, biểu mẫu được điền bằng trận Test đầu tiên theo thứ tự dữ liệu, không lọc theo việc dự đoán đúng/sai.
2. Muốn đổi trận, chọn gameId rồi bấm **Nạp trận mẫu**. Danh sách chứa đủ 988 trận Test.
3. Bấm **Dự đoán kết quả**. Xem đội dự đoán thắng, xác suất hai đội và chênh lệch vàng/XP/mạng của đầu vào đã gửi.
4. Tích **Hiện kết quả thực tế của trận mẫu** để kiểm tra đúng/sai.
5. Thử sửa vàng hoặc kinh nghiệm và bấm dự đoán lại. Web coi đây là tình huống giả định, không lấy nhãn trận gốc chấm đúng/sai.
6. Bấm **Xóa để nhập mới** nếu muốn tự nhập đủ 25 thông số độc lập. Các cột dẫn xuất được tính tự động.

Dữ liệu trong form được gửi khi bấm nút. Kết quả bên dưới luôn ghi rõ là **lần gửi gần nhất**; sửa ô nhập chưa làm kết quả cũ tự cập nhật. Khi điều hướng sang trang khác, bản nháp được giữ theo lần gửi gần nhất; hãy bấm dự đoán trước khi rời trang nếu muốn lưu các giá trị vừa chỉnh trong phiên.

### Dự đoán CSV

- Tải CSV mẫu trực tiếp trên web hoặc dùng `samples/input_template.csv`.
- CSV phải dùng UTF-8 (có hoặc không có BOM), dấu phẩy phân cột, dấu chấm cho phần thập phân.
- Nhận 1–10.000 dòng, dung lượng tối đa 10 MB. Đây là giới hạn vận hành của ứng dụng.
- Bắt buộc đủ 25 cột nêu ở bảng bên dưới. File gốc đủ 38 cột thống kê cũng hợp lệ; 13 biến dư thừa nếu được cung cấp phải nhất quán với công thức gốc.
- `gameId` là tùy chọn, được giữ để đối chiếu, không vào mô hình. Nhãn `blueWins` cũng tùy chọn, chỉ dùng sau suy luận.
- Cột bổ sung ngoài schema được bỏ qua và thông báo tên. Thiếu biến bắt buộc, sai kiểu, số âm, số đếm không nguyên, NaN hoặc vô cực sẽ chặn dự đoán và nêu dòng/cột lỗi.
- Nhãn đối chiếu sai không làm mất dự đoán hợp lệ; web báo không tính được chỉ số đối chiếu. Nhãn chỉ có một lớp thì ROC-AUC hiển thị không xác định.
- Kết quả theo lô có `prob_blue_win`, `prob_red_win`, `prediction` (1=Blue, 0=Red), `predicted_winner`, cùng `gameId` nếu có. Với nhãn hợp lệ, thêm `actual_blueWins` và `correct`.
- Khi đổi file, kết quả của file trước được xóa khỏi giao diện để tránh nhầm lẫn.

| Nhóm | Các cột bắt buộc |
|---|---|
| Kinh tế | blueTotalGold, redTotalGold |
| Giao tranh | blueKills, redKills, blueAssists, redAssists |
| First Blood | blueFirstBlood |
| Kinh nghiệm/cấp độ | blueTotalExperience, redTotalExperience, blueAvgLevel, redAvgLevel |
| Lính/quái rừng | blueTotalMinionsKilled, redTotalMinionsKilled, blueTotalJungleMinionsKilled, redTotalJungleMinionsKilled |
| Mục tiêu | blueDragons, redDragons, blueHeralds, redHeralds, blueTowersDestroyed, redTowersDestroyed |
| Mắt | blueWardsPlaced, redWardsPlaced, blueWardsDestroyed, redWardsDestroyed |

First Blood, Dragons và Heralds hỗ trợ 0/1 theo schema của mô hình hiện tại. `blueFirstBlood=0` nghĩa là Red giành First Blood; không biểu diễn trường hợp chưa có First Blood. Cấp độ trung bình nằm trong 1–18. Các số đếm là số nguyên không âm. Không áp min/max của dữ liệu Train như ràng buộc cứng: giá trị ngoài khoảng quan sát chỉ được cảnh báo.

### Xem kết quả nghiên cứu

- **Kết quả thực nghiệm**: số trận, tỷ lệ nhãn, chia tập, Test metrics, cấu hình mô hình, so sánh Validation, confusion matrix, ROC, importance và ablation.
- **Phân cụm & luật**: K, Silhouette, hồ sơ cụm, Elbow/PCA và luật với Support/Confidence/Lift, có nút tải luật.
- Các số liệu đọc từ CSV/JSON thực nghiệm, không giả lập. Nhánh phân cụm chỉ hiển thị kết quả đã lưu, không gán cụm cho trận mới.

## 4. Kết quả đã tái lập

Dữ liệu: 9.879 trận, 40 cột. Chia Train 7.903, Validation 988, Test 988; seed 42. Mô hình được chọn: Histogram Gradient Boosting, 25 đặc trưng, `learning_rate=0.05`, `max_leaf_nodes=7`, `l2_regularization=1`, `max_iter=150`, `early_stopping=False`; ngưỡng 0,5. Refit trên 8.891 trận Train + Validation.

| Test metric | Giá trị |
|---|---:|
| Accuracy | 0,7226720648 |
| Precision · Blue | 0,7176938370 |
| Recall · Blue | 0,7322515213 |
| F1 · Blue | 0,7248995984 |
| ROC-AUC | 0,7963120044 |

K=2, Silhouette mẫu 0,2583426382; 584 tập mục phổ biến độ dài tối đa 3 và 75 luật sau lọc. Các số liệu chính khớp báo cáo Word tại mức làm tròn được trình bày. Tốc độ fit thay đổi theo máy, không phải chỉ số chất lượng mô hình.

Xác suất dự đoán của từng trận không phải Accuracy. Chênh lệch vàng/XP/mạng là mô tả đầu vào, không chứng minh nhân quả hoặc thay thế phương pháp giải thích mô hình.

## 5. Đưa lên Streamlit Community Cloud

Hướng dẫn đối chiếu tài liệu chính thức ngày 21/10/2026:

- [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

### A. Đưa code lên GitHub

1. Đăng nhập tài khoản GitHub của bạn, tạo repository mới, ví dụ `lol-10min-prediction`.
2. Giải nén ZIP trước. Đưa **nội dung bên trong** thư mục `LoL_Streamlit` lên repository để `app.py` và `requirements.txt` ở thư mục gốc. Không chỉ tải lên tệp ZIP.
3. Giữ nguyên các thư mục `results/`, `figures/`, `samples/`, `data/` và `.streamlit/`. File ẩn `.streamlit/config.toml` có thể tạo bằng nút Add file → Create new file nếu trình tải lên không chọn được.
4. Không đưa `.venv/`, `__pycache__/` hoặc thông tin đăng nhập lên GitHub. Dự án này không yêu cầu secrets.
5. Kiểm tra có `results/final_model.joblib`, hai file Python chính và hai file trong `samples/`.

### B. Tạo ứng dụng

1. Mở https://share.streamlit.io/ và đăng nhập/kết nối GitHub của bạn.
2. Chọn **Create app**, rồi **Yup, I have an app** nếu giao diện hỏi.
3. Chọn repository vừa tạo và branch thực tế, thường là `main`.
4. Ở **Main file path**, nhập `app.py` (nếu bạn tải nguyên thư mục lên thì phải dùng đường dẫn tương ứng, ví dụ `LoL_Streamlit/app.py`).
5. Mở **Advanced settings**, chọn **Python 3.12**. Để trống Secrets.
6. Đặt **App URL** là một tên còn khả dụng, chẳng hạn `lol-10min-ten-cua-ban`. Đây chỉ là ví dụ tên đề xuất, chưa được đăng ký.
7. Bấm **Deploy** và chờ cài thư viện/nạp ứng dụng.
8. Sau khi hệ thống báo thành công, dùng URL `.streamlit.app` mà Streamlit thực sự cung cấp.

### C. Kiểm tra sau triển khai

- Mở ứng dụng, dự đoán một trận mẫu, hiện nhãn thực tế.
- Đổi thông số rồi dự đoán lại, kiểm tra thông báo tình huống giả định.
- Tải lên CSV mẫu và tải kết quả xuống.
- Mở trang kết quả thực nghiệm và phân cụm/luật.
- Thử trên điện thoại hoặc thu hẹp cửa sổ trình duyệt.
- Nếu cần gửi đường dẫn cho người khác, kiểm tra thiết lập chia sẻ của app.

Không thay đổi phiên bản scikit-learn/numpy tùy tiện sau khi đã lưu joblib. Môi trường huấn luyện và suy luận cần tương thích; bản giao khóa đúng các phiên bản đã dùng.

## 6. Chạy lại kiểm thử và thực nghiệm

Kiểm thử web và suy luận (không huấn luyện):

```bash
python tests/test_app.py
```

Kết quả kiểm thử lưu tại `results/web_verification.json`. AppTest kiểm tra ứng dụng và các widget; đối tượng upload được đưa vào hàm uploader trong test để chạy toàn bộ nhánh xử lý CSV. Thao tác chọn file của trình duyệt chưa được kiểm tra end-to-end trên trình duyệt thật.

Chỉ khi muốn tái tạo thực nghiệm từ đầu:

```bash
python league_of_legends_10min_data_mining.py --data data/high_diamond_ranked_10min.csv
```

Lệnh này ghi đè các kết quả thực nghiệm và mô hình trong `results/`, ảnh trong `figures/`. Khi giữ nguyên dữ liệu, seed và cấu hình thì các trận mẫu vẫn phù hợp. Nếu thay dữ liệu hoặc cách chia tập, phải tạo lại `samples/test_matches.csv` theo `split_assignments.csv`, CSV mẫu và `input_ranges_train.csv` theo Train mới trước khi công bố. Bản web hiện được bàn giao cho đúng dữ liệu và mô hình gốc, không tự tái tạo các tệp demo khi bạn tự thay thực nghiệm.

## 7. Khắc phục lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không có lệnh streamlit | Dùng `python -m streamlit run app.py` trong đúng môi trường đã cài requirements |
| Thiếu final_model.joblib | Giải nén/tải lên đủ `results/`; không đổi tên mô hình |
| ModuleNotFoundError | Cài lại `python -m pip install -r requirements.txt` bằng đúng Python đang chạy web |
| Lỗi tương thích joblib/sklearn | Chọn Python 3.12 và cài đúng phiên bản trong requirements; dùng mô hình đi kèm |
| CSV không đọc được | Xuất CSV UTF-8 phân cách dấu phẩy, không dùng XLSX hoặc dấu chấm phẩy |
| Có số liệu âm/sai cột | Xem bảng lỗi dòng/cột trên web; đối chiếu CSV mẫu |
| Không hiện kết quả mới sau sửa | Bấm lại nút Dự đoán để gửi form |
| Deploy báo thiếu thư mục | Kiểm tra đường dẫn tương đối, chữ hoa/thường trên Linux và cấu trúc repository |
| Hình không hiện | Tải lên đủ `figures/` và giữ tên PNG |

## 8. Ghi chú kiểm chứng và phạm vi

Web không có lệnh fit/huấn luyện trong luồng tương tác. Mô hình được cache bằng `st.cache_resource`; bảng dữ liệu được cache bằng `st.cache_data`. Dự đoán dùng đúng thứ tự `feature_names` và ngưỡng trong artifact. Khóa dùng chung bảo vệ thao tác suy luận trong nhiều phiên; threadpool giới hạn ở 2 để giữ mức sử dụng CPU phù hợp.

CSV người dùng tải lên được xử lý trong bộ nhớ phiên và không được code web ghi ra đĩa. Không hỗ trợ người truy cập tải lên mô hình joblib/pickle. Không kết nối tài khoản Riot, không thu thập trận đấu trực tiếp, không yêu cầu mật khẩu hoặc API key.

Báo cáo Word không bị chỉnh sửa. Có thể bổ sung riêng một mục “Ứng dụng minh họa” với ảnh giao diện và kịch bản demo sau khi bạn mở web hoặc triển khai thành công.

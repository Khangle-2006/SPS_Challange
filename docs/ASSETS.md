# Dữ liệu, checkpoint và bundle tái lập

Repo chứa mã và số liệu tổng hợp, không cấp quyền sử dụng hay phân phối lại
dataset. Không có tự động download nhãn, bảng nhân khẩu học hoặc holdout.

**Có gói tải:** [checkpoint + cấu hình reader](INFERENCE_ASSETS.md).
Dataset trên server: [SERVER_DATASETS.md](SERVER_DATASETS.md).

**BTC joint hiện hành:** danh mục riêng cho fit/replay, schema mảng và lệnh tạo
manifest SHA256 nằm ở [BTC_JOINT_REPRODUCING.md](BTC_JOINT_REPRODUCING.md).
Các mục `numeric_dev` và `reader_v3` bên dưới thuộc archive; chúng không thay thế
bundle `export_extracted`/native/shifted/reference mà lệnh BTC joint yêu cầu.

## Lấy từ đâu

| Tài sản | Nguồn / cách yêu cầu | Cần nhận và kiểm |
|---|---|---|
| UBFC-rPPG Dataset 2 | [Trang chính thức của tác giả](https://sites.google.com/view/ybenezeth/ubfcrppg) | Video `subject*/vid.avi`, `ground_truth.txt`; xin thêm split dev 35 người của đội để lặp đúng bảng, không dùng 7 người test |
| MCD | Xin maintainer đường dẫn nguồn và điều khoản đúng **phiên bản MCD đội đã dùng**; repo chưa xác minh link tải công khai | View FullHD, PPG, mapping session/person, timestamp retained-frame và receipt đồng hồ. Không thay bằng dataset khác cùng tên |
| BTC dev | Cổng dữ liệu của ban tổ chức theo quyền của đội; nhờ maintainer cấp **dev-only** bundle | 218 video được chấm, 158 người; nhãn PR, person/fold, activity, metadata nguồn và danh sách loại. Không kèm 72 video/41 người holdout |
| AiPEX | Theo nguồn/điều khoản ban tổ chức; không cần cho inference reader_v4/P4m | Chỉ cần cho tái lập nhánh huấn luyện lịch sử, không thuộc quick start |
| Ba expert | [open-rppg 0.1.1 trên PyPI](https://pypi.org/project/open-rppg/0.1.1/) | Cài profile experts; package chứa weights. So SHA bằng smoke test; không tải weights từ link lạ |
| CNN padding-fixed của đội | [Release inference-assets-r1-20261124](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124) | 5 bản xuất chỉ giữ model/epoch, đúng fold 0–4 seed 2027; tensor giữ nguyên, SHA file xuất và nguồn có trong manifest |
| Reader và cache nghiên cứu | Reader BTC cho CLI nằm trong `models/`; Release R1 chứa đúng deploy/fold/P4m hiện hành; split/cache vẫn xin maintainer | Chỉ tham số được công khai; per-video predictions/labels giữ riêng. Không trộn model CLI với phổ expert chưa reset của lượt cũ |

Không thay checkpoint fold0 cho tất cả fold. Các file readout đóng băng có thể
chứa cả nhãn và dự đoán từng cửa sổ (ví dụ `z14_p4m_frozen.npz`); **không** đưa
nguyên file vào GitHub Release công khai. Release trên chỉ trích các tham số cần cho suy luận và khóa hash riêng; không
phát hành các mảng chứa nhãn/ID/prediction của file gốc.

Hai weights open-rppg đã được ghi trong định nghĩa đội:

```text
FacePhys.rlap.weights.h5  6cd99510bcc5771f9cebcdb0f47a5cb3692d19f1e3d6e2444af5e8fac0ee3c07
ME.pure.weights.h5        2135995cb4161d1d4c7c3aa45d333dc6b1f5818b0652d11da9fc0fb0b6e11bf0
```

ME-chunk.pure và ME-flow.pure dùng cùng file weights, khác chế độ xử lý. License
mã không tự chứng minh mọi quyền đối với weights/dữ liệu huấn luyện; xem
`THIRD_PARTY_NOTICES.md` và điều khoản upstream trước khi phân phối lại.

## Checklist gửi maintainer

Yêu cầu bundle cho stage cụ thể, hệ điều hành/Python và commit của repo. Không
đưa dữ liệu cá nhân, thông tin đăng nhập hoặc video lên issue công khai. Gói tái lập bảng cần:

1. `inference_repair_20261113/MANIFEST.json`, `FROZEN.json` và các reference,
   source clock/cache mà chúng trỏ tới; scope chỉ dev công khai/BTC dev.
2. Numeric BTC `w2_btc_20261116/numeric_dev/VIDEOS.json`, `blocks/*.npz`,
   các bảng baseline đã dùng; private recording mapping (y05/y11 cần).
3. `claude_research_20261118/out/dev_video_meta.csv` và dữ liệu phụ thuộc của
   `lib_btc`/reader; dùng metadata **nguồn**, không dùng kích thước crop/constraint.
4. Crop/phổ các stage được chọn (`out/x10`, `x15`, `y01_spectra.pkl`,
   public x13/x20/y12/y13...), cùng completion/clock/hash receipts.
5. Năm checkpoint public và model reader đúng phạm vi OOF/deploy; không fit trên
   người validation và không dùng model all-dev để báo điểm dev OOF.
6. Danh mục file đầy đủ, SHA256 và `complete_for` (stage nào có đủ dependency).

Những stage chỉ chạy từ cache không nhất thiết cần toàn bộ video raw. Bundle
phải ghi rõ điểm bắt đầu và bảng nào tái lập được; không gọi cache partial là
bundle đủ mọi bảng.

## File kê khai riêng

Dùng `configs/assets.example.json` làm mẫu **ngoài Git**, ví dụ
`../private-assets/assets.json`. Mỗi file có:

```json
{
  "schema_version": 1,
  "scope": "dev-only",
  "complete_for": ["btc-reader-v3"],
  "files": [
    {
      "source": "received/VIDEOS.json",
      "destination": "reports/w2_btc_20261116/numeric_dev/VIDEOS.json",
      "sha256": "<64 ký tự hex do maintainer cung cấp>",
      "kind": "metadata",
      "relocate_json_paths": false
    }
  ]
}
```

Ví dụ trên chỉ minh họa **một** entry, chưa đủ để chạy stage. `source` tương đối
với thư mục chứa manifest. `destination` tương đối với workspace và không được
chứa `..`, đường dẫn tuyệt đối hoặc ghi đè code. Liệt kê **từng file**; không
trỏ cả thư mục raw gồm holdout. SHA phải đến từ bundle được cấp quyền; không
điền hash giả để doctor PASS.

Chỉ JSON có đường dẫn máy chủ mới cần `relocate_json_paths=true`. Không đổi
nhãn, fold, hoạt động hay các hash tham chiếu frozen; hash lỗi phải được xử lý
bằng receipt có nguồn gốc rõ, không sửa âm thầm. Không deserialize pickle từ
nguồn không tin cậy; các script lịch sử có dùng pickle và chỉ nhận artifact
được cấp quyền, kiểm hash từ maintainer.

`tools/repro.py prepare` chép nguyên binary, ghi receipt hash trước/sau metadata
đã đổi đường dẫn, không xóa nguồn. Cấu hình riêng, assets, outputs và venv được
gitignore; luôn kiểm `git status` trước commit.

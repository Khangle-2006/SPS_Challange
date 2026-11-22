# Vị trí dataset trên P100

Các đường dẫn dưới đây đã kiểm tra tồn tại ngày 23/11/2026. Đây là đường dẫn
trên **server P100**, không phải link tải Internet và không xuất hiện sau `git
clone`. Cần tài khoản SSH/server được chủ dữ liệu cấp quyền.

| Dataset | Đường dẫn thật trên P100 |
|---|---|
| BTC / FaceVitals | `/mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107` |
| UBFC-rPPG | `/data_pool/khangle/facevitals/datasets/UBFC-rPPG` |
| MCD-rPPG | `/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG` |
| AiPEX | `/data_pool/khangle/facevitals/datasets/AiPEX` |

`/data_pool/khangle/facevitals/datasets/MCD-rPPG` là symlink sang đường MCD trên
Samsung ở bảng. Không tải/chép lại dataset nếu đang làm trên cùng server.

Thư mục dataset có quyền đọc/ghi rộng, nhưng các thư mục cha `/data_pool` và
`/data_pool/khangle` hiện chỉ cho owner/nhóm `datapool` đi qua. Nếu gặp
`Permission denied`, nhờ quản trị viên cấp tài khoản/quyền nhóm phù hợp;
đường dẫn tồn tại không đồng nghĩa mọi tài khoản đều truy cập được. Lượt đóng
gói này không thay đổi quyền server.

## Chỉ chạy tập dev được phép

Root dataset chứa dữ liệu gốc, **không phải danh sách được phép chấm**. Không
dùng `find`/glob toàn bộ root để xây tập test. Dùng đúng manifest dev và split
đã khóa; BTC holdout 72 video/41 người và 7 người test UBFC2 vẫn đóng.

Các hồ sơ nội bộ cần nhận từ maintainer khi tái lập:

| Mục đích | Vị trí nội bộ |
|---|---|
| BTC dev manifest | `/mnt/samsung_t5/khangle/facevitals/reports/round1_v4_20261105/btc_zeroshot/dev_manifest.json` |
| Cửa sổ public/BTC và fold của audit | `/mnt/samsung_t5/khangle/facevitals/reports/round1_execution_20261110/inference_repair_20261113/MANIFEST.json` |
| Cache/phổ thí nghiệm constraint | `/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints/out` |

Các manifest/cache có thể chứa nhãn hoặc ID; không đưa chúng lên GitHub. Bundle
checkpoint/reader công khai chỉ chứa trọng số và tham số, không thay thế các
hồ sơ nội bộ này. Muốn dùng máy khác, yêu cầu chủ dữ liệu cấp bản dev hợp lệ,
sau đó cấu hình đường dẫn theo `docs/GETTING_STARTED.md`.

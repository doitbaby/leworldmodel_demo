# Kế hoạch train LeWorldModel trên FPT AI Factory

> Repo: [`doitbaby/leworldmodel_demo`](https://github.com/doitbaby/leworldmodel_demo) (toàn bộ 7 milestone M1–M7 đã merge vào `main`)
> Budget: **$100 USD**
> Mục tiêu: train một checkpoint LeWM thật (không phải smoke) trên H100/H200, đo benchmark 5-mode, mang `best.pt` về máy local để Unity sidecar dùng.

---

## TL;DR (Bản tóm tắt 1 đoạn)

Dùng **AI Notebook** (Jupyter, 1-click, persistent storage) của FPT AI Factory với **1× H100 SXM5 ($2.31/giờ)**. Pipeline 5 bước: (1) đăng ký tài khoản tại `id.fptcloud.com` + liên hệ FPT sales kích hoạt AI Factory subscription, (2) tạo Notebook H100, clone repo, `pip install -r tools/lewm/requirements.txt`, (3) upload gameplay JSONL v3 (record từ Unity) qua Jupyter UI hoặc `wget`/`scp`, (4) chạy `python -m tools.lewm.train --device cuda --observation-mode board-jsonl --epochs 30 --val-split 0.1 --lr-schedule cosine` (~30 phút wall clock cho dataset vài ngàn transitions), (5) chạy `python -m tools.lewm.benchmark --checkpoint results/lewm/best.pt --episodes 20 --modes random mission mlp_lite lewm_no_planner lewm_dreamer` + download `best.pt` (~vài MB) qua Jupyter UI. **Chi phí ước tính: $5–$15 cho lần đầu**, **$100 thừa cho 10+ lần thử nghiệm + hyperparam sweep**.

---

## 1. Mục tiêu & Acceptance criteria

| # | Tiêu chí | Cách verify |
| --- | --- | --- |
| 1 | Train được checkpoint LeWM thật (không phải smoke) trên dữ liệu Unity gameplay JSONL v3 | `results/lewm/best.pt` tồn tại, file size ~1–5 MB, `metrics.csv` có ≥20 epoch với `val_loss` giảm |
| 2 | Benchmark 5-mode trên RogueSimEnv (M6) bằng ckpt vừa train | `benchmark.summary.csv` có `lewm_dreamer.mean_levels_cleared > mission.mean_levels_cleared > random.mean_levels_cleared` |
| 3 | Chi phí ≤ $100 (còn dư để retry/sweep) | FPT Cloud billing dashboard |
| 4 | `best.pt` mang về local Unity được, `WorldModelPlannerAgent` (M5) load qua sidecar OK | Unity Editor → Play → toggle `UseSidecar=on` → BrainHUD hiển thị `WORLD MODEL SIDECAR (Nx H)` |

---

## 2. FPT AI Factory landscape (kết quả research)

FPT AI Factory cung cấp 4 cách dùng GPU, mình sắp xếp theo độ phù hợp với LeWM port:

| Sản phẩm | Cách dùng | GPU đơn lẻ | Giá H100 80GB | Phù hợp cho LeWM? |
| --- | --- | --- | --- | --- |
| **AI Notebook** | 1-click JupyterLab, persistent storage, UI upload file | 1× H100 hoặc CPU | **$2.31/giờ** (per second) | ⭐ **KHUYẾN NGHỊ** — zero infra setup, đủ mạnh, có persistent workspace |
| **GPU Container** | Docker container có sẵn 6 template (Jupyter, PyTorch CUDA, Ubuntu...), SSH, ports | 1×–8× H100/H200 | $2.54/giờ (per second) | OK — chỉ chọn nếu cần custom Docker image hoặc multi-port |
| **GPU Virtual Machine** | Full Ubuntu VM + Jump Server, SSH key, public IP | 1×–8× H100/H200 | $2.54/giờ (per second) | Hơi nặng — cần tự cài driver, set Security Group, mở port; **chỉ chọn nếu muốn chạy luôn sidecar inference trên cloud** |
| **Bare Metal** | Server vật lý 8× H100/H200 dedicated | 8× cứng | (theo hợp đồng) | KHÔNG hợp — overkill cho ~280K-param model |
| **Model Finetuning** | Managed pipeline (chỉ cho foundation models có sẵn) | (theo pipeline) | $5.5/GPU-giờ | KHÔNG hợp — không cho custom code |

**Bonus**: FPT cũng có **B300** (Blackwell Ultra, 288 GB GPU mem) đang ở pre-order, không cần đến cho 280K-param model — H100 80GB là quá dư.

### 2.1 Billing logic (quan trọng)

- **Billed per-second** (round per 15 phút). Nghĩa là nếu mình tắt notebook ngay sau khi train xong, không tốn tiền chờ.
- Storage persistent: ~$0.00013/GB/giờ → 100 GB × 24 giờ × 7 ngày = $0.22/tuần. **Negligible**.
- **Low balance protection**: dưới 1 giờ cost → bị limit create/start container. Phải nạp đủ trước.
- **Negative balance**: container tự dừng. Dữ liệu trên temporary storage **mất**, persistent storage giữ. Sau 7 ngày → xoá hết. ⚠️

### 2.2 Yêu cầu đăng ký

1. Tạo account tại <https://id.fptcloud.com/>
2. **Liên hệ FPT sales** để kích hoạt subscription "AI Factory – AI Infrastructure" (theo doc, đây là requirement riêng, không tự self-serve được).
3. Region: chỉ có **Hanoi-2** (Việt Nam) hoặc **Tokyo** (Nhật). Mặc định nên chọn Hanoi-2 cho latency thấp + tiền VND.
4. Nạp $100 vào balance.

---

## 3. Kiến trúc data flow (cách dữ liệu đi từ Unity → cloud → Unity)

```
┌────────────────────────────────┐       ┌──────────────────────────────────┐
│ Local Windows máy của bạn      │       │ FPT AI Factory (Hanoi-2)        │
│ (Unity 6000.4.6f1)             │       │                                  │
│                                │       │  ┌────────────────────────────┐  │
│  Unity Editor                  │       │  │ AI Notebook (1× H100)      │  │
│   ↓ Play, gameplay loop        │       │  │                            │  │
│  RogueTransitionRecorder.cs    │       │  │  /workspace/                │  │
│   ↓ writes JSONL v3            │       │  │   leworldmodel_demo/        │  │
│  rogue_transitions.jsonl       │       │  │    ├── tools/lewm/...       │  │
│   (vài MB → vài chục MB)       │       │  │    └── data/                │  │
│                                │       │  │        └── rogue_*.jsonl    │  │
│   ─── upload via Jupyter UI ──>│ ─────>│  │                            │  │
│       hoặc `wget` từ Drive     │       │  │   results/lewm/             │  │
│                                │       │  │    ├── best.pt              │  │
│  Unity Editor                  │       │  │    ├── checkpoint.pt        │  │
│   ↑ load best.pt via sidecar   │ <─────│  │    ├── metrics.csv          │  │
│  python -m tools.lewm.serve    │       │  │    └── benchmark.{,summary} │  │
│   --checkpoint best.pt         │       │  │                            │  │
│                                │       │  │  python -m tools.lewm.train │  │
│  WorldModelPlannerAgent        │       │  │  python -m tools.lewm.benchmark
│   ↓ POST /plan_actions         │       │  └────────────────────────────┘  │
└────────────────────────────────┘       └──────────────────────────────────┘
```

**Lưu ý quan trọng**: Sidecar (`python -m tools.lewm.serve`) chạy trên **máy local** chứ không phải trên cloud — chỉ training mới cần GPU. Inference trên CPU đủ nhanh cho 1 query mỗi decision tick.

---

## 4. Lộ trình thực hiện (8 bước)

### Bước 1 — Account & subscription (1 lần, mất ~1 ngày làm việc nếu cần FPT sales duyệt)

1. Đăng ký <https://id.fptcloud.com/> bằng email + xác minh.
2. Vào FPT Cloud Console, gửi yêu cầu kích hoạt "AI Factory – AI Infrastructure" (theo doc cần contact sales).
3. Nạp $100 vào balance (qua thẻ visa/master hoặc chuyển khoản VNĐ).
4. Set **Low Balance Alert** ngưỡng $20 để không bị bất ngờ.

### Bước 2 — Chuẩn bị dữ liệu local (1 lần, ~30 phút)

Trên máy Windows có Unity:

1. Mở `Assets/Scenes/WorldModelPlannerRoom.unity` → Play.
2. Chơi 20–50 episode (mix: 10 ép mission-only + 10 ép random + 10 ép thử-sai). `RogueTransitionRecorder` ghi ra `$HOME/.config/unity3d/{Company}/{Product}/rogue_transitions.jsonl`.
3. Verify schema:
   ```powershell
   Get-Content $HOME\.config\unity3d\...\rogue_transitions.jsonl | Select-Object -First 1
   # phải thấy "schema":"rogue.transition.v3" và có "board_state" / "next_board_state"
   ```
4. Đếm transitions:
   ```powershell
   (Get-Content $HOME\.config\unity3d\...\rogue_transitions.jsonl).Count
   # mục tiêu ≥ 5000 transitions (~50 episode × 100 steps)
   ```
5. Zip lại → upload Google Drive (hoặc S3) để có URL public download trên cloud.

### Bước 3 — Tạo AI Notebook (~2 phút)

1. FPT Cloud Console → **AI Factory → AI Notebook** → **Create Notebook**.
2. Chọn flavor: **1× H100 (15 CPU, 250 GB RAM, 80 GB VRAM)** — $2.31/giờ.
3. Persistent storage: **50 GB** (đủ cho code + JSONL + checkpoints, ~$0.16/tuần).
4. Region: **Hanoi-2**.
5. Đợi 1–2 phút → notebook ready, click **Open JupyterLab**.

### Bước 4 — Setup repo trong Notebook (~5 phút, một lần)

Mở Terminal trong JupyterLab:

```bash
cd /workspace
git clone https://github.com/doitbaby/leworldmodel_demo.git
cd leworldmodel_demo

# venv không cần (notebook đã có Python 3.10+ + pip)
pip install -r tools/lewm/requirements.txt

# verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# kỳ vọng: True NVIDIA H100 80GB HBM3
```

### Bước 5 — Smoke test trên GPU (~1 phút, $0.04)

Chạy ngay smoke training để confirm pipeline CUDA OK trước khi training thật:

```bash
python -m tools.lewm.train --smoke --device cuda
# kỳ vọng: epoch 1 loss 1.768 -> epoch 2 loss 1.435 (giống VM CPU nhưng nhanh ~10x)
```

Nếu fail → check `torch.__version__` + `torch.version.cuda` matches H100 (cần CUDA 12+).

### Bước 6 — Upload data + Train thật (~20–60 phút, $0.8–2.5)

```bash
mkdir -p /workspace/leworldmodel_demo/data
# Option A: upload qua Jupyter UI (kéo-thả file zip)
# Option B: wget từ Google Drive public link
wget "https://drive.google.com/uc?export=download&id=<FILE_ID>" -O data/rogue_transitions.jsonl.zip
unzip data/rogue_transitions.jsonl.zip -d data/

cd /workspace/leworldmodel_demo

# Train chính:
python -m tools.lewm.train \
    --device cuda \
    --observation-mode board-jsonl \
    --jsonl-path data/rogue_transitions.jsonl \
    --epochs 30 --batch-size 128 \
    --val-split 0.1 \
    --lr-schedule cosine \
    --warmup-steps 200 --min-lr-ratio 0.05 \
    --metrics-csv results/lewm/metrics.csv \
    --output results/lewm/checkpoint.pt \
    --best-output results/lewm/best.pt
```

**Theo dõi**:
- `metrics.csv` cập nhật mỗi epoch: theo dõi `val_loss` giảm liên tục.
- `nvidia-smi` chạy parallel tab để kiểm GPU util — nên ≥80%, nếu thấp tăng `--batch-size`.

**Ước thời gian** (1× H100):
- ~5000 transitions × 30 epoch × batch 128 = ~1200 batches × 30 epoch = ~36000 steps.
- H100 fp32 throughput cho model ~280K params: ~1–3 ms/step → **~2–10 phút wall clock**.
- Cộng IO + JSONL parse + checkpoint save: **20–30 phút** an toàn.

### Bước 7 — Benchmark 5-mode (~10–20 phút, $0.4–0.8)

```bash
python -m tools.lewm.benchmark \
    --episodes 20 --max-steps 200 --seed 0 \
    --checkpoint results/lewm/best.pt \
    --modes random mission mlp_lite lewm_no_planner lewm_dreamer \
    --output-csv results/lewm/benchmark.csv
```

Open `results/lewm/benchmark.summary.csv` trong Jupyter và verify acceptance criterion #2.

### Bước 8 — Download artifacts + Stop notebook (~1 phút)

1. JupyterLab → right-click `results/lewm/best.pt` → **Download** (~1–5 MB).
2. Cũng tải `metrics.csv` + `benchmark.summary.csv` để báo cáo.
3. **CRITICAL**: FPT Cloud Console → AI Notebook → **Stop** notebook để dừng billing GPU. Persistent storage vẫn giữ data (~$0.16/tuần) cho lần sau.

---

## 5. Code/file deltas cần (mostly không cần đổi gì)

| File | Cần đổi? | Lý do |
| --- | --- | --- |
| `tools/lewm/train.py` | **Không** | đã có `--device` flag từ M1, `.to(device)` ở M1/M4 |
| `tools/lewm/requirements.txt` | **Không** | `torch>=2.1` auto-detect CUDA |
| `tools/lewm/data.py` | **Không** | JSONL parsing pure Python |
| `tools/lewm/jepa.py` | **Không** | mọi tensor đều respect `device` qua module hierarchy |
| `tools/lewm/benchmark.py` | **Không** | hỗ trợ `--device cuda` qua arg (kế thừa train.py) |

**Đề xuất 2 PR nhỏ (optional, không block training)**:

1. **PR `M8a: GPU smoke test in CI`** — thêm 1 job CI optional `gpu-smoke` chạy trên `[self-hosted, gpu]` runner nếu user có self-hosted, hoặc skip mặc định. Không bắt buộc; chỉ giúp tránh regression khi đổi `train.py` device handling.

2. **PR `M8b: training script for FPT AI Factory`** — thêm file `tools/lewm/scripts/train_fpt_h100.sh` đóng gói toàn bộ Bước 5–7 thành 1 script, chỉ cần `bash train_fpt_h100.sh <data_path>`. Sẽ hữu ích nếu user train nhiều lần.

→ **Đề xuất**: Để mình làm PR M8b sau khi user xác nhận đã train thành công lần đầu, để không over-engineer trước khi biết pipeline chạy OK.

---

## 6. Bảng chi phí ước tính

| Hoạt động | GPU-giờ | USD |
| --- | --- | --- |
| Smoke test (verify CUDA) | 0.02 | $0.05 |
| Train chính (lần 1, ~30 ép, 30 epoch) | 0.5 | $1.16 |
| Benchmark 5-mode (20 ép/mode) | 0.3 | $0.69 |
| **Tổng lần 1 (happy path)** | **~0.8** | **~$2** |
| Hyperparam sweep × 5 (LR, embed_dim, sigreg_weight) | 4 | $9.24 |
| Tinh chỉnh + retry × 5 | 5 | $11.55 |
| Storage 50 GB × 2 tuần | — | $0.30 |
| **Tổng "ambitious" budget** | **~10 giờ** | **~$25** |

→ **$100 dư cho ~40 giờ H100**, đủ cho mọi kịch bản hợp lý. Nếu chỉ train 1 lần ngon → **chỉ tốn ~$2**.

---

## 7. Risks & Mitigations

| Rủi ro | Khả năng | Mitigation |
| --- | --- | --- |
| FPT sales duyệt subscription mất 1+ ngày | Med | Liên hệ sales ngay khi user đọc plan này. Trong lúc chờ, record JSONL data từ Unity local. |
| Region Hanoi-2 hết H100 stock | Low | Fallback Tokyo region (giá tương đương, latency cao hơn ~50ms với VN). |
| Notebook timeout / disconnect giữa training | Med | Dùng `nohup` hoặc `tmux` trong terminal: `nohup python -m tools.lewm.train ... > train.log 2>&1 &`. Notebook persistent storage giữ checkpoints. |
| `torch.cuda.is_available()` = False | Low | FPT notebook đã preinstall CUDA driver. Nếu False → khả năng pip cài bản torch CPU-only, fix: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124` |
| JSONL data quá ít, val_loss không giảm | Med | Trước khi train, in `len(dataset)`. Nếu < 1000 transitions → đi thu thập thêm gameplay. Synthetic mode (`--observation-mode synthetic`) là fallback để verify pipeline còn ngon. |
| Quên stop notebook → billing burn | Low | Set **Low Balance Alert** ở $20. Mỗi 8 giờ idle H100 mất $18.5 nên tổng cùng lắm $20 nếu quên 1 đêm. |
| Bị deleted sau 7 ngày negative balance | Low | $100 budget với ~$2.5/giờ = không vào negative trong vài chục giờ runtime. |
| Upload JSONL chậm qua Google Drive | Low | JSONL nhỏ (vài MB). Có thể `scp` qua SSH (GPU Container option) nếu cần. |

---

## 8. Verification plan

Sau khi user hoàn thành Bước 8, mình sẽ help verify:

1. **Sanity check ckpt**: `python -c "import torch; ckpt=torch.load('best.pt'); print(ckpt.keys(), ckpt['config'])"` — phải thấy keys `model_state`, `config`, schema `lewm.port.checkpoint.v1`.
2. **Sidecar load test trên local**: `python -m tools.lewm.serve --checkpoint best.pt --host 127.0.0.1 --port 5555` → `curl http://127.0.0.1:5555/info` → phải trả về `embed_dim`, `action_dim=4`, `max_horizon`, etc.
3. **Benchmark sanity**: open `benchmark.summary.csv`, verify acceptance criterion #2 (`lewm_dreamer > mission > random` theo `mean_levels_cleared`).
4. **End-to-end Unity** (user làm): Unity Editor → Play → `UseSidecar=on` → bấm M → BrainHUD hiển thị `WORLD MODEL SIDECAR (Nx H)`, agent không đi vào enemy, không stuck wall.

---

## 9. Fallback plan (nếu FPT AI Factory không khả thi)

Trường hợp gặp blocker (sales chậm duyệt, region hết stock, billing không cho phép), các option backup theo thứ tự cost-effectiveness:

1. **Vast.ai** — H100 spot ~$1.5–2.5/giờ, instant deploy, không cần subscription duyệt tay. Trade-off: data ít tin cậy hơn (host người dùng cá nhân), không có SLA.
2. **Runpod** — H100 secure cloud $2.79/giờ, community $1.99/giờ. UI tốt hơn FPT, có ssh + jupyter sẵn.
3. **Lambda Cloud** — H100 $2.49/giờ. US-only, latency cao từ VN nhưng pipeline giống hệt.
4. **Google Colab Pro+** ($50/tháng) — A100 40GB. Đủ cho model 280K params, free-tier có thể đủ smoke. Trade-off: idle disconnect, không persistent storage trừ khi mount Drive.
5. **Kaggle Notebooks** — free P100 16GB. ĐỦ cho 280K-param model. 30-giờ/tuần limit. Zero cost.

→ Nếu user vẫn muốn ép FPT (budget đã có $100), khuyến nghị **kiên nhẫn chờ sales 1–2 ngày** + record data Unity trong lúc đó.

---

## 10. Open questions (cần user trả lời để mình tinh chỉnh plan)

> Mình **chưa code/setup gì** cho đến khi có câu trả lời, vì 1–2 quyết định dưới đây sẽ thay đổi pipeline đáng kể.

1. **Account status**: bạn đã đăng ký `id.fptcloud.com` chưa? Đã contact FPT sales kích hoạt AI Factory subscription chưa? (Nếu chưa → đây là step 1 trước khi mọi thứ khác.)
2. **Data**: bạn đã có file `rogue_transitions.jsonl` (schema v3) chưa, hay vẫn cần record từ Unity? Nếu có rồi, file bao nhiêu transitions / size?
3. **Tier sản phẩm**: ưu tiên **AI Notebook** (Jupyter, dễ nhất) hay **GPU Container** (Docker template, SSH, cheaper per-second nếu dùng nhiều lần) hay **GPU VM** (full Ubuntu + Jump Server, nếu muốn chạy luôn sidecar inference trên cloud)?
4. **Mức độ involvement**: bạn muốn mình:
   - (a) Chỉ giao plan này, bạn tự setup + train + báo kết quả về,
   - (b) Mở PR thêm `tools/lewm/scripts/train_fpt_h100.sh` đóng gói toàn bộ pipeline thành 1 lệnh, hoặc
   - (c) Bạn cho mình SSH vào FPT notebook để mình chạy training & gửi `best.pt` về?

---

**Khi bạn trả lời 4 câu trên, mình sẽ confirm cost estimate cuối + bắt đầu execute đúng option bạn chọn.**

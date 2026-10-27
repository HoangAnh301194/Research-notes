# P1 — Hướng dẫn chạy notebook một lần để bổ sung metric và keyframe (27/10/2026)

**Không cần retrain**. Sử dụng lại checkpoint VideoMamba hiện tại. Mục tiêu: (1) tái sử dụng CSV/cache nếu có, (2) tính thêm metric offline bằng CPU, (3) trích TXT + ảnh lỗi keyframe, (4) chạy inference bổ sung cho **Accuracy@10%, @20%, @50%** khi thực sự cần.

Tài liệu P0: [P0_SOTA_AUDIT_20261027.md](P0_SOTA_AUDIT_20261027.md). Script P1: [p1_offline_metrics_keyframes.py](p1_offline_metrics_keyframes.py).

## A. Bước 0: kiểm tra artifact có sẵn trước khi bật GPU

**Tìm trước trong Kaggle Notebook Output/Dataset cũ**, không chỉ xem session Kaggle hiện tại. Tên file đúng code:

| Dataset | Isolated CSV | Continuous cache | Sinh ở cell |
|---|---|---|---|
| IPN-Hand | `/kaggle/working/videomamba_ipn_isolated_test.csv` | `/kaggle/working/videomamba_postprocess_raw_cache.pkl` | Isolated **cell index 12** (Section 4), Continuous **cell 18** (Section 5.2) |
| CTPGesture v1 | `/kaggle/working/ctpgesture_videomamba_evaluation/isolated_predictions.csv` | `/kaggle/working/ctpgesture_videomamba_evaluation/continuous_raw_cache.pkl` | Isolated **cell 16** (Section 6), Continuous **cell 22** (Section 7.2) |

**Nếu output cũ còn:** tải về hoặc đính kèm Kaggle Dataset chứa các file vào phiên mới, sau đó dùng script CPU bên dưới ngay. Đối với IPN, `continuous_raw_cache` chứa `P_raw`, `window_starts/ends`, `gt_labels`; đối với CTP cũng có các trường đó. **Không chạy lại Section 5.2/7.2** nếu cache đúng checkpoint và thông số vẫn còn.

**Nếu phải suy luận lại:** kiểm tra đúng checkpoint path, split/label, class mapping, sample frames, version của code, native CUDA extension. Không đổi kiến trúc, preprocessing, batch tùy tiện rồi gộp kết quả thành cùng một thí nghiệm.

**CTP quan trọng:** Trong config cell index 6, `REUSE_PER_VIDEO_CACHE=True` có nguy cơ đọc cache cũ nếu đổi checkpoint/config. Khi tạo thí nghiệm mới sau khi thay model/config, đặt `REUSE_PER_VIDEO_CACHE=False` và dùng `OUTPUT_DIR` mới; chỉ bật lại khi đã kiểm tra SHA-256 checkpoint, 32/8 window, 16 sample frames, preprocessing và class mapping trùng chính xác. `RUN_DEBUG_VIDEO=False` tránh chạy lặp 1 video.

## B. Lệnh chạy phân tích P1 CPU (không forward model)

Trong Kaggle mở Internet nếu cần clone repo GitHub công khai; hoặc upload riêng file `p1_offline_metrics_keyframes.py` vào Kaggle. Sau khi có script trên máy:

```bash
# CTPGesture — path script giả định clone Research-notes vào /kaggle/working
python /kaggle/working/Research-notes/Research_Report/261022/p1_offline_metrics_keyframes.py \
  --dataset ctp \
  --isolated /kaggle/working/ctpgesture_videomamba_evaluation/isolated_predictions.csv \
  --cache /kaggle/working/ctpgesture_videomamba_evaluation/continuous_raw_cache.pkl \
  --video-root /kaggle/input \
  --out /kaggle/working/p1_ctp \
  --top 30

# IPN-Hand
python /kaggle/working/Research-notes/Research_Report/261022/p1_offline_metrics_keyframes.py \
  --dataset ipn \
  --isolated /kaggle/working/videomamba_ipn_isolated_test.csv \
  --cache /kaggle/working/videomamba_postprocess_raw_cache.pkl \
  --video-root /kaggle/input \
  --out /kaggle/working/p1_ipn \
  --top 30
```

Nếu đã clone repo phiên khác hoặc script ở path khác thì chỉnh lại **đúng vị trí file**. Nếu chỉ có isolated CSV, bỏ `--cache ...`; nếu chỉ có raw cache, bỏ `--isolated ...`. Cache là `pickle`: **chỉ đọc cache do chính cậu hoặc nhóm tạo**, không tải pickle của người lạ.

Output gồm:
- `isolated_metrics.json`, `isolated_classification_report.csv`, `isolated_confusion_matrix.csv`.
- `isolated_misclassified_segments.csv`: danh sách clip sai, GT/pred/confidence; `error_keyframes.txt`.
- `keyframe_error_montages/`: ảnh ghép frame **10%, 50%, 90%** từ các clip bị dự đoán sai (nếu tìm thấy video); **đây là representative frames cho debug, không phải attention keyframe do mô hình tự chọn**.
- `continuous_raw_per_video_extra_metrics.csv`, `continuous_raw_metrics_extra.json`: Accuracy, Macro-F1, Balanced Accuracy, edit distance và mean per-video.
- `continuous_hard_frame_segments.csv`, `continuous_error_keyframes.txt`, `keyframe_error_montages_continuous/`.
- `summary.json` tổng hợp.

**Cảnh báo metric:** `levenshtein_gt_normalized_pct` trong script chỉ là **chỉ số thử nghiệm** với công thức `100×(1−edit_distance/len(GT gesture sequence))`. Cần kiểm tra chính xác cách xây chuỗi/chuẩn hóa với **official IPN-Hand Levenshtein protocol** trước khi công bố cạnh benchmark. Bảng RSC causal có **Online-F1 và TTC**; không thể tính tương đương chỉ từ offline cache vì dự đoán của notebook dùng các cửa sổ có dữ liệu tương lai.

## C. Nếu phải chạy lại notebook, thứ tự cell tối thiểu

### C.1 IPN-Hand

File: [261007/videomamba_man_configsBenchmark.ipynb](../261007/videomamba_man_configsBenchmark.ipynb).

- Chạy lần lượt **cell index 2 → 3 → 5 → 7 → 9 → 10 → 12 (isolated) → 14 (continuous functions) → 18 (all 52 videos)**; index tính từ **0** trong cấu trúc ipynb. Các Markdown cell không cần chạy.
- Cell 5: `NUM_FRAMES=16`, `WINDOW_SIZE=64`, `WINDOW_STEP=16`, `THETA_SEC=2.0`; kiểm tra `CHECKPOINT_PATH` và `df_test` gắn với official annotations.
- Cell 12 có `ISOLATED_INCLUDE_NO_GESTURE=False`, `ISOLATED_MAX_SAMPLES=None`, `ISOLATED_SAVE_CSV=True`.
- Cell 18: `ALL_MAX_VIDEOS=None`, `SAVE_RAW_POSTPROCESS_CACHE=True`, `ALL_SAVE_CSV=True`. Nếu chỉ bổ sung metric từ cache, **không chạy** cell này.
- Cell 20 chạy postprocessing ablation từ cache, không cần forward; **không chọn ngưỡng tối ưu trên test rồi báo như independent test**.
- Cell 22–30 benchmark kiến trúc VideoMAE rất tốn thời gian cài/import, **không cần chạy lại cho P1** nếu chỉ thiếu metric/keyframe.

### C.2 CTPGesture v1

File: [261015/VideoMambaTinyCTPGestureFinal.ipynb](../261015/VideoMambaTinyCTPGestureFinal.ipynb).

- Chạy lần lượt **cell index 2 → 4 → 6 → 8 → 10 → 12 → 14 → 16 (isolated) → 18 (continuous functions) → 22 (all 11 test videos)**.
- Cell 6: `TEST_IDS=["002","004","008","010","012","014","016","018","101","102","104"]`, `NUM_FRAMES=16`, `WINDOW_SIZE=32`, `WINDOW_STEP=8`. `ISOLATED_MAX_SAMPLES=None`, `CONTINUOUS_MAX_VIDEOS=None`, `RUN_DEBUG_VIDEO=False`. Tạo `OUTPUT_DIR` riêng nếu có bất kỳ thay đổi checkpoint/config.
- Cell 16 tạo `isolated_predictions.csv` **đã chứa `prob_0` tới `prob_8`** nên đủ để phân tích confidence/misclassification mà **không infer**.
- Cell 22 tạo `continuous_raw_cache.pkl` (chứa full window softmax probabilities/GT) và `continuous_raw_per_video.csv`.
- Cell 24 là hậu xử lý từ cache; cell 26 benchmark FLOPs/latency; **không chạy hai cell này** nếu mục tiêu chỉ là metrics/keyframe từ output đã lưu.

## D. Thí nghiệm mới cần GPU: Accuracy@10%, @20%, @50% trên **GT isolated segments**

**Định nghĩa để báo cáo:** Cấp VideoMamba **chỉ các frame từ đầu đoạn gesture đến mốc 10%, 20%, 50%** của GT segment, sau đó dùng lại phép uniform-sample 16 frames trong đoạn prefix đó. Nếu prefix có ít hơn 16 frame, thuật toán lấy mẫu có thể **lặp frame**; ghi rõ trong Methods. Đây là **recognition of isolated prefixes with known start boundary**, không phải real-time gesture spotting. Không đặt nhãn metric này bằng **RSC Early@10 (10 output steps)**.

### D.1 Cell thêm **sau IPN cell index 12**

```python
# IPN: chỉ cần model và isolated_results của notebook hiện tại
import numpy as np, pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

IPN_EARLY_DIR = Path("/kaggle/working/p1_ipn/early")
IPN_EARLY_DIR.mkdir(parents=True, exist_ok=True)
early_summary = []
for frac in (0.10, 0.20, 0.50):
    output = IPN_EARLY_DIR / f"isolated_prefix_{int(frac*100)}.csv"
    if output.exists():
        old = pd.read_csv(output)
        if len(old) == len(isolated_results):
            print("Reusing", output)
            df = old
        else:
            output.unlink()
            df = None
    else:
        df = None
    if df is None:
        rows = []
        for r in tqdm(isolated_results.itertuples(index=False),
                      total=len(isolated_results), desc=f"Prefix {frac:.0%}"):
            start, end = int(r.t_start), int(r.t_end)
            prefix_end = min(end, start + max(1, int(np.ceil((end-start+1)*frac))) - 1)
            frames, sampled = extract_frames_from_video(
                r.video_path, start, prefix_end, NUM_FRAMES
            )
            pred, conf, probs = run_inference(model, frames, DEVICE)
            rows.append(dict(video=r.video, t_start=start, t_end=end,
                             prefix_end=prefix_end, gt_idx=int(r.gt_idx),
                             pred_idx=int(pred), confidence=float(conf),
                             correct=(int(pred)==int(r.gt_idx))))
        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
        print("Saved", output)
    y = df["gt_idx"].astype(int)
    p = df["pred_idx"].astype(int)
    early_summary.append(dict(observed_fraction=frac, clips=len(df),
                              accuracy=accuracy_score(y,p),
                              macro_f1=f1_score(y,p,average="macro",zero_division=0)))
pd.DataFrame(early_summary).to_csv(IPN_EARLY_DIR/"accuracy_at_prefix.csv",index=False)
display(pd.DataFrame(early_summary))
```

**Không dùng `ISOLATED_INCLUDE_NO_GESTURE=True` nếu cậu muốn đối chiếu với kết quả isolated 1.101 samples hiện có.** Tránh thay split/sample list giữa các mức prefix.

### D.2 Cell thêm **sau CTP cell index 16**

```python
# CTP: tái sử dụng hàm run_isolated_video_sequential() trong notebook
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

EARLY_DIR = OUTPUT_DIR / "p1_early"
EARLY_DIR.mkdir(parents=True, exist_ok=True)
early_summaries = []
for frac in (0.10, 0.20, 0.50):
    dest = EARLY_DIR / f"isolated_prefix_{int(frac*100)}.csv"
    if dest.exists():
        df = pd.read_csv(dest)
        if len(df) != len(isolated_df):
            raise ValueError(f"Incomplete output: {dest}, back up & remove explicitly")
        print("Reuse", dest)
    else:
        parts = []
        prefix_df = isolated_df.copy()
        length = prefix_df["t_end"].astype(int)-prefix_df["t_start"].astype(int)+1
        prefix_df["t_end"] = np.minimum(
            prefix_df["t_end"].astype(int),
            prefix_df["t_start"].astype(int) + np.maximum(1,np.ceil(frac*length).astype(int))-1
        )
        prefix_df["duration_frames"] = prefix_df["t_end"] - prefix_df["t_start"] + 1
        for vid, vdf in tqdm(list(prefix_df.groupby("video_id",sort=False)),
                             desc=f"CTP {frac:.0%}"):
            records, decode_sec, forward_sec = run_isolated_video_sequential(
                model, vdf, ISOLATED_BATCH_SIZE
            )
            parts.extend(records)
        df = pd.DataFrame(parts)
        df.to_csv(dest,index=False)
        print("Saved",dest)
    y = df["gt_idx"].astype(int)
    p = df["pred_idx"].astype(int)
    early_summaries.append(dict(prefix_fraction=frac, segments=len(df),
                                accuracy=accuracy_score(y,p),
                                macro_f1=f1_score(y,p,average="macro",zero_division=0),
                                gesture_gt_only_accuracy=accuracy_score(
                                    y[y!=0],p[y!=0])))
pd.DataFrame(early_summaries).to_csv(EARLY_DIR/"accuracy_at_prefix.csv",index=False)
display(pd.DataFrame(early_summaries))
```

**Ghi chú chuyên môn:** CTP có **869 no-gesture GT segments**. Báo cáo cả `9-class all-segments` và `GT gesture-only 859 samples` để tránh việc kết quả @10/@20/@50 bị chi phối bởi các đoạn non-gesture. **Không** xem sự tăng giảm giữa các mức prefix là chứng cứ online TTC/early command recognition nếu chưa chuyển sang causal continuous protocol.

## E. Bảo toàn output và điều kiện nghiệm thu

1. **Trước khi kết thúc phiên Kaggle**, download ZIP toàn bộ `p1_ipn/` hoặc `p1_ctp/`, CSV, raw cache, checkpoint manifest; **Kaggle working không phải lưu trữ vĩnh viễn giữa sessions**. Save Version và đối chiếu tab Output để giữ artifact.
2. Lưu `SHA-256` checkpoint, Git commit/nb version, số lượng test samples, test video IDs, FPS, frames/model, window/stride, native CUDA mode và model evaluation flag.
3. Kiểm tra `confusion_matrix.sum() == len(isolated_results)`, `continuous raw per-video frame` bằng notebook gốc. Không công bố số metric nếu pipeline P1 không tái hiện Raw Accuracy.
4. Đảm bảo ảnh được trích từ **đúng video và frame index**: **IPN isolated annotation 1-based**, **CTP continuous/isolated 0-based**.
5. Xuất bảng Accuracy@10/20/50, Macro-F1, error TXT, ít nhất 10 contact sheets cho lỗi có confidence cao, kiểm tra bằng mắt GT/pred. Mỗi hình cần có ID, GT, predicted class, start/end, frame indices.
6. Để so fair SOTA: test dùng để **báo cáo**, không dùng để tune threshold rồi lại báo như unseen test; cần validation tách riêng ở cấp video nếu muốn chọn ngưỡng hậu xử lý.

### Kết quả mong đợi

- **P1 offline:** Metrics và keyframe audit hoàn tất mà **không cần chạy lại VideoMamba** nếu còn CSV/cache và video gốc.
- **P1 prefix inference:** Có Accuracy@10/20/50 và Macro-F1 tại từng mức cho cả IPN/CTP, không cần train model.
- **P1 mở rộng về sau (nếu cần):** thiết kế evaluator *causal end-of-window* và Online-F1/TTC theo định nghĩa công trình RSC 2026; đây là **thí nghiệm khác** so với prefix isolated.

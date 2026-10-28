# P1 metric protocol and qualitative visualization options — 28/10/2026

## CURRENT IMPLEMENTATION — 16 exact sampled frames, all cases by default

Both new standalone P1 notebooks now export **two distinct figure protocols**, and the full evaluation remains a checkpoint-only rerun. No original notebook is changed.

1. **Isolated (all evaluated clips):** each gesture-annotated clip produces a 4×4 mosaic of **exactly 16 model temporal input frames**, preprocessed with the matching spatial Resize(224) + CenterCrop(224) before visualization. Export correct and incorrect samples to separate JPEG folders, with complete per-sample metadata, source frame indices, decode-failure audit and same-GT-class nearest-duration error/control pairing CSV. Output: `isolated_16frame/{correct,error}/*.jpg` and `isolated_all_samples_index.csv`. IPN annotation indices are 1-based; CTP are 0-based. The CSV stores source indices zero-based for unambiguous frame retrieval.

2. **Continuous (all GT gestures + all unmatched predicted gestures):** match GT gesture segments to predicted gesture segments *one-to-one* at temporal IoU≥0.50. Each matched pair is a **correct** case; unmatched GT segments and unmatched predictions are **errors** (different roles are identified in the case manifest; one physical failure may create both an unmatched GT and an unmatched prediction record). For each event, select the **actual model inference window** having greatest overlap with the event, extract that window's **exact 16 sampled input frames** using `build_sample_offsets`, and append a GT-versus-RAW-Prediction timeline. Save the indices of ALL windows contributing to the event and to its midpoint, so the representative window is never misrepresented as the only contributing window. Output: `continuous_16frame_timeline/{correct,error}/*.jpg` and `continuous_all_events_index.csv`. Background-only true-negative frame intervals are not expanded into one image per frame; framewise predictions remain available in the RAW cache.

**Export defaults:** `P1_ISO_LIMIT=None` and `P1_CONT_LIMIT=None` mean **100% of relevant samples/events**. Display only two image previews per folder to keep Kaggle output compact; all exported JPEGs are still saved. Exported image count, disk usage and time depend on the actual number and lengths of samples. The notebook continues recording decode failures if individual frames are unavailable.

**Important scientific scope:** The 16 frames are exact *temporal input samples* and depict the matching spatial crop before pixel normalization, NOT model attention-selected salient keyframes. Continuous predictions aggregate overlapping windows and can use future frames, so they are OFFLINE and do not imply causal TTC/Online-F1.

### P1 extended diagnostics (no extra inference)

- **Isolated:** Top-2/Top-3 Accuracy and Balanced Accuracy at observation ratios 10/20/50/75/100%; per-class precision, recall, F1; first correctly classified ratio; first *stably* correct ratio over the evaluated ratio grid; count of high-confidence ≥0.90 wrong predictions.
- **Continuous:** TP/FP/FN and precision/recall/F1 at temporal IoU 0.10/0.25/0.50; per-class segment-level F1@50; number of predicted versus GT gesture segments; unmatched predictions per minute; miss rate; matched correct-class onset/offset signed and absolute errors in seconds, conditioned on one-to-one tIoU≥0.50.
- **Files:** `extra_isolated_metrics/{topk_and_balanced_accuracy_by_ratio.csv,early_correctness_per_sample.csv,early_correctness_summary.json,isolated_per_class_metrics.csv}`, `extra_continuous_metrics/{per_video_detection_precision_recall.csv,per_class_segment_f1_at_50.csv,matched_onset_offset_errors.csv,detection_boundary_summary.json}`.
- **Reference basis:** MS-TCN CVPR 2019 (Edit/F1@10/25/50): https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html ; ASRF WACV 2021 (boundary/oversegmentation): https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html ; IPN-Hand original official benchmark: https://gibranbenitez.github.io/IPN_Hand/Results.html ; Guo et al. ICML 2017 calibration: https://proceedings.mlr.press/v70/guo17a.html .

**Validation status:** GitHub notebooks structurally parsed and source cells verified; Kaggle dataset/CUDA execution not yet performed. Compare original accuracy and segment F1 with P1 and inspect example frames to certify exact temporal/spatial sampling before quoting numbers in a paper.

## Scope and evidence

This P1 update creates standalone copies of the checkpoint-based evaluation notebooks; it does not modify the original notebooks, training or checkpoint weights. No Kaggle execution or numerical result is claimed yet.

- IPN **new P1 notebook**: `Research_Report/KaggleNotebook/VideoMamba_IPN_Hand_P1_Metrics_Keyframes_Python313.ipynb`
- CTP **new P1 notebook**: `Research_Report/KaggleNotebook/VideoMamba_CTPGesture_P1_Metrics_Keyframes_Python313.ipynb`
- **Original notebooks are unmodified:** `Research_Report/261007/videomamba_man_configsBenchmark.ipynb` and `Research_Report/261015/VideoMambaTinyCTPGestureFinal.ipynb`.
- The standalone notebooks contain P1 cells immediately after the original *isolated* and *full continuous* evaluation cells. Unrelated one-video debug, architecture comparison and postprocessing ablation sections have been removed, so **run top-to-bottom** once the GPU, datasets, original checkpoints and new verified wheel Dataset are attached.
- For CTP P1, `REUSE_PER_VIDEO_CACHE=False` is set to avoid mixing predictions from different checkpoints/configurations. The output directory is also isolated from the original evaluation.

## Metric definitions and applicability

| Metric | Evaluated on | Protocol | Reference |
|---|---|---|---|
| Top-1 Accuracy, Macro-F1 | Isolated GT annotated gesture clips | Original official test subset; CTP also reports 9-class and GT-gesture-only | IPN-Hand benchmark (ICPR 2020): https://gibranbenitez.github.io/IPN_Hand/Results.html |
| Accuracy@10/20/50/75/100% | Isolated GT gesture prefixes | Use first ceil(ratio * full_segment_length) frames and uniformly sample exactly 16 model frames. Full-length result reused. This assumes **known GT start/end boundaries** and is NOT online detection. | Fernando & Herath, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Fernando_Anticipating_Human_Actions_by_Correlating_Past_With_the_Future_With_CVPR_2021_paper.html |
| Normalized accuracy AUC | Accuracy over observation ratios | Trapezoidal integral on [10%,100%], divided by 0.9; not the full [0%,100%] AUC | ERA, ECCV 2022: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940657.pdf |
| Frame Accuracy, Macro-F1 | Continuous raw frame labels | Raw overlapping-window probability average; **offline** | MS-TCN, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html |
| Segmental F1@10/25/50 | Continuous raw gesture segments | One-to-one correct-class segment matching, temporal IoU thresholds 0.10/0.25/0.50; class 0 excluded | MS-TCN CVPR 2019, MS-TCN++ IEEE TPAMI 2023: https://pubmed.ncbi.nlm.nih.gov/32886607/ |
| Edit Score | Continuous raw gesture sequence | 100*(1−Levenshtein(segments_GT, segments_pred)/max(number_segments_GT, number_segments_pred,1)), then average per video | MS-TCN, CVPR 2019 |
| IPN GT-normalized Levenshtein diagnostic | Continuous IPN | 100*(1−edit_distance/max(1, number_of_GT_gestures)); mean per video; MUST separately verify exact benchmark detector+recognizer protocol before comparing official leaderboard | Original IPN leaderboard: https://gibranbenitez.github.io/IPN_Hand/Results.html |
| ECE (15 bins), multi-class Brier, NLL | Isolated raw softmax | Fixed bins; **no test-set calibration/tuning**. Brier uses sum over all K class errors (not divided by K) | Guo et al., ICML 2017: https://proceedings.mlr.press/v70/guo17a.html |

**Do not label the isolated observation-ratio curve as RSC-GestureNet Early@10.** RSC Early@10 means a correct command within ten *causal output steps* after a gesture transition; TTC means latency to first correct emitted command. Their 2026 PRCV CTPGesture study uses causal full-frame input and a distinct online protocol. It cannot be directly compared against the current offline VideoMamba 32/8 or 64/16 evaluation: https://github.com/chengli24/rsc-gesturenet-prcv2026

## Generated files

IPN root: `/kaggle/working/p1_ipn_re_evaluation/`
CTP root: `/kaggle/working/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/`

- `isolated_prefix_010.csv`, `020.csv`, `050.csv`, `075.csv`, `100.csv`: individual sample GT, prediction, confidence and softmax at each observation ratio.
- `early_accuracy_curve.csv`, `early_auc.json`: observation ratio metrics and normalized AUC.
- `isolated_calibration.json`: ECE, NLL, multiclass Brier.
- `continuous_raw_segment_metrics_per_video.csv`, `continuous_raw_segment_metrics_summary.json`: frame/segment sequence metrics.
- `isolated_error_cases.csv`, `isolated_correct_cases.csv`, `continuous_error_cases.csv`, `continuous_correct_cases.csv`: manifest for visualization, including video ID/path, clip boundaries and class labels.

Save a Kaggle version and/or export files after the run. Original video files are still needed to render frame images. IPN isolated annotations are **1-based**, while CTP frames are **0-based**. The IPN continuous manifest explicitly uses 1-based indices for easy alignment with the annotation files.

## Visualization choices — choose before rendering

**A. Static contact sheet (isolated, easiest paper figure).** Extract the original 16 sampled input frames from *one* GT segment. Heading: video ID, start/end, GT class, predicted class, softmax confidence. Put a red border on errors and a neutral/green border on correct controls. Recommended when studying posture/motion recognition failure.

**B. Matched error vs correct cases (strongest controlled qualitative comparison).** Two rows of equally spaced frames: wrong and correct samples with the **same GT class**, as close as practical in gesture length and recording/viewpoint; record matching criteria. Avoid claiming model mechanisms from the image alone. Recommended for paper qualitative examples.

**C. Continuous prediction timeline (best for temporal errors).** Per video segment, combine three horizontal tracks: representative frames at timestamps, GT frame labels, and RAW model predictions. Highlight class confusions, missed gestures, false detections, delayed boundaries, and over-segmentation. Show actual duration/fps. Since RAW offline windows observe future frames, do not call the timeline causal/online.

**D. Failure taxonomy gallery (most useful for debugging).** Grid grouped by false positive (background→gesture), false negative (gesture→background), gesture-class confusion, and temporal boundary/fragmentation. Include correct reference examples per class.

**E. Probability trajectory beside contact sheet (diagnostic).** Show class probability curves for the top 2–3 classes from continuous raw aggregated per-frame probabilities, with GT labels and segment boundaries. For isolated-prefix samples, show confidence vs observation ratio rather than a genuine per-frame probability trace.

**Recommendation:** Use **B + C** for the paper; use **D** for engineering debugging and failure analysis. A is a quick first implementation. Only original representative video frames are available; **none of these is an attention-derived learned keyframe** without additional attribution analysis.

## Acceptance checks before publishing numbers

1. Confirm checkpoint SHA, dataset split/test video IDs, 16-frame preprocessing, window/stride, seed and native CUDA/FP32 regime are unchanged from the intended comparison.
2. Confirm prefix result row counts match isolated baseline, and 100% accuracy exactly matches the original isolated result.
3. Confirm global continuous RAW accuracy equals the notebook's existing RAW metric within numerical tolerance.
4. Confirm IPN/CTP frame indexing before inspecting visual frames; inspect ≥10 misclassified and ≥10 correctly predicted examples.
5. Test the new cells on Kaggle (unavailable in this coding environment). Do not treat unexecuted cells and generated CSV filenames as validated experimental results.
6. Avoid choosing thresholds on the sealed test split; reserve validation videos for any model/postprocessing selection.

## Native wheel and Python 3.13 compatibility gate

The new notebooks discover a **validated** builder `wheel_manifest.json` under `/kaggle/input`. They check current Python wheel tag (e.g., cp313), full PyTorch version, PyTorch CUDA runtime, C++11 ABI, GPU SM targets, original VideoMamba pinned source commit, wheel SHA-256 and the builder's CUDA smoke-test status. A source checkout, native CUDA activation, and dummy model forward are required before the costly P1 evaluation begins. If any check fails, the run stops with a diagnostic message; it does not silently reinstall cp312 wheel or benchmark with an accidental slow backend.

This is **compatibility checking**, not proof that the user's NEW wheel Dataset is already attached or that the entire notebook has executed successfully on Kaggle. Attach the two native wheels and manifest and confirm runtime logs. If the dataset lacks `wheel_manifest.json`, publish the manifest exported by `EgoEvGesture_VideoMamba_CUDA_Wheel_Builder_Kaggle_EN_v1.ipynb` with the new pair.

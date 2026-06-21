# Experiment Log

Use this file as the running lab notebook. Keep entries short, dated, and reproducible.

## Template

### YYYY-MM-DD - Experiment name

**Question:**  
What did this experiment test?

**Setup:**  
Dataset, image size, model, scheduler, training steps, hardware.

**Result:**  
Main metric values and qualitative observations.

**What worked:**

**What failed / looked suspicious:**

**Next step:**

\---

## 2026-05-30 - Repo scaffold

**Question:**  
What is the initial research framing and project structure?

**Setup:**  
Created v0.1 repository scaffold with README, project spec, roadmap, technical report skeleton, and placeholder source modules.

**Result:**  
Ready to begin dataset selection and implementation.

**Next step:**  
Choose initial public-domain film sources and implement dataset extraction.



## 2026-05-30 - Dataset pilot: Nosferatu first 20 minutes

**Question:**
Can the opening 20 minutes of *Nosferatu* provide a usable pilot corpus for archival restoration experiments?

**Setup:**
Source film: *Nosferatu* (1922), Internet Archive item `Nosferatu_DVD_quality`.
Source file: `nosferatu-1of5_512kb.mp4`.
Segment: 00:00:00 to 00:20:00.
Sampling: one frame every two seconds.
Preprocessing: grayscale, resized to width 512 pixels.
Expected output: approximately 595 raw frames.

**Result:**
The extracted frame set is suitable for pipeline validation, but contains many intertitles/title cards. These need to be removed before training or degradation ranking.

**What worked:**
The first 20-minute segment provides a manageable pilot corpus with visible archival texture, high-contrast cinematography, and varied scenes.

**What failed / looked suspicious:**
A large number of frames are title cards or text-heavy intertitles. These would bias the restoration task toward text reconstruction rather than image restoration.

**Next step:**
Apply OCR-assisted filtering to separate text/title frames from non-text cinematic frames.

---

## 2026-05-31 - OCR-based title-card filtering

**Question:**
Can OCR filtering remove intertitles/title cards more reliably than a visual heuristic based on brightness, contrast, and connected components?

**Setup:**
Input directory: `data/raw_frames/nosferatu_1of5_20min_fps0_5/`.
OCR method: EasyOCR, English text detection.
Output directories: non-text frames copied to `data/selected_frames/nosferatu_non_text_easyocr/`; OCR-flagged text frames copied to `data/selected_frames/nosferatu_text_easyocr/`.
Audit outputs: CSV saved to `evaluations/results/nosferatu_easyocr_text_filter.csv`; visual review sheet saved to `evaluations/figures/nosferatu_easyocr_flagged_text_frames.png`.

**Result:**
The OCR filter produced a visibly better separation of title cards from cinematic frames than the earlier heuristic. Initial visual review of the flagged text-frame sheet suggests that identified frames are highly clustered around intertitle sequences, which matches expected behaviour. 168 frames removed from analysis (logged in evaluations/results/nosferatu_easyocr_text_filter.csv)

**What worked:**
OCR-based filtering better matches the actual exclusion criterion: visible text. Keeping a separate text-frame directory also makes the exclusion auditable.

**What failed / looked suspicious:**
OCR may still miss stylised or low-confidence intertitles, and may occasionally flag non-title cinematic frames containing signage or high-contrast shapes. The flagged-frame contact sheet should be retained for audit.

**Next step:**
Use the EasyOCR-filtered non-text directory as the input for degradation ranking. Produce clean-ish and naturally degraded candidate sets, each with contact sheets for manual review.

---

## 2026-05-31 - Degradation ranking and candidate set creation

**Question:**  
Can non-text *Nosferatu* frames be sorted into relatively clean archive candidates and naturally degraded stress-test candidates using simple image-quality heuristics?

**Setup:**  
Input directory: `data/selected_frames/nosferatu_non_text_easyocr/`.  
Ranking method: combined degradation score based on high-frequency residual, Laplacian variation, scratch-like vertical edge features, and exposure penalty.  
Outputs: ranking CSV saved to `evaluations/results/nosferatu_degradation_ranking.csv`; clean-ish candidates copied to `data/curated/nosferatu_cleanish_candidates/`; degraded candidates copied to `data/curated/nosferatu_degraded_candidates/`; contact sheets saved to `evaluations/figures/`.

**Result:**  
Visual review of clean-ish and degraded contact sheets denotes reasonably strong sorting.

**What worked:**  
Degraded content has inconsistent colouring, darker frames, motion blur, and visible blemishes (e.g. white circle for 000385. Clean-ish content still contains some blur and darkness (particularly around frame edges), but tend to be stylistic and intentional.

**What failed / looked suspicious:**  
Some clean-ish frames are still quite blurry, and visible blemishes in 000392, 000443, 000449-000457, and 000595. These have been moved to degraded folder manually.

**Next step:**  
Use the clean-ish set for synthetic blemish generation and the naturally degraded set as the real archival restoration stress test.

## 2026-05-31 - Synthetic blemish example and project visual

**Question:**  
Can we create a simple visual explanation of the project’s two evaluation regimes: controlled synthetic restoration and naturally degraded archival stress testing?

**Setup:**  
Input clean-ish candidates: `data/curated/nosferatu_cleanish_candidates/`.  
Input degraded candidates: `data/curated/nosferatu_degraded_candidates/`.  
Synthetic degradation script: `scripts/create_synthetic_blemish_examples.py`.  
Visualisation script: `scripts/create_three_use_case_visual.py`.  
Synthetic pair output: `data/processed/synthetic_pairs/nosferatu_cleanish/`.  
Figure output: `outputs/figures/archive_diffusion_three_use_cases.png`.

**Result:**  
Created a synthetic restoration example consisting of an artificially blemished input, its original clean-ish target, and a real naturally degraded archival frame for comparison. Combined these into a three-panel figure showing the core experimental design.

**What worked:**  
The figure makes the distinction between synthetic fidelity evaluation and real archival restoration clear. The synthetic pair provides a known target for future PSNR/SSIM-style metrics, while the naturally degraded example motivates qualitative blemish-removal and authenticity-preservation evaluation.

**What failed / looked suspicious:**  
The synthetic blemish looks quite digitally artificial; grain is clear and scratches are ok but the circles are very circular. Consider revisiting this in a future step.

**Next step:**  
Embed the figure in the technical write-up, then begin formalising the evaluation metrics and rubrics for synthetic fidelity, real blemish removal, and over-smoothing/authenticity preservation.

## 2026-06-01 - Baseline synthetic restoration evaluation

**Question:**  
How do simple non-diffusion restoration baselines perform on the synthetic archival blemish benchmark?

**Setup:**  
Created a synthetic restoration dataset from clean-ish *Nosferatu* frames. Each target frame was resized to 128×128 grayscale and degraded using scratch/dust, grain, and blur/contrast transformations. Baselines included unchanged input, median filtering, OpenCV inpainting/denoising, and non-local means denoising.

**Result:**  
Pending review of `evaluations/results/baseline_restoration_summary.csv` and `outputs/evaluation_grids/baseline_restoration_examples.png`.

**What worked:**  
The evaluation harness now produces per-example metrics, grouped summary metrics, and qualitative grids using a consistent prediction-manifest format. It demonstrates that the synthetic tasks have different difficulty profiles - for grain, median is strong, for scratch dust, mask-based inpainting is strong (requiring access to a mask), and for blur contrast, all the baselines are pretty weak. A good diffusion model should: 

* Beat median on grain, or preserve more structure at similar denoising quality
* Beat/informatively compare with inpainting on scratch_dust
* Beat all simple baselines on blur_contrast

**What failed / looked suspicious:**  
Pending. Key checks: whether synthetic blemishes look plausibly archival, whether inpainting is unfairly advantaged by masks, and whether denoising baselines over-smooth film texture.

**Next step:**  
Use the same synthetic dataset and evaluation harness for the first conditional diffusion model.

## 2026-06-06 - First conditional DDPM restoration model

**Question:**  
Can a compact conditional DDPM learn to restore synthetically degraded archival film crops when conditioned on the degraded input?

**Setup:**  
Training data: `data/processed/synthetic_restoration/nosferatu_v0/`.  
Image size: 128×128 grayscale.  
Conditioning: noisy target concatenated with degraded input as two input channels.  
Model: compact U-Net diffusion model using a DDPM scheduler.  
Training objective: predict Gaussian noise added to the clean target image.  
Evaluation: same synthetic restoration metrics as the classical baselines.
Trained on CPU

**Result:**  
Model ran and training loss improved over the limited number of runs (5). Generated restoration grid, and summary metrics.

**What worked:**  
Model ran end-to-end and training loss fell.

**What failed / looked suspicious:**  
Model outputs significantly underperformed baseline, both visually in the restoration grid and in terms of summary metrics.

**Next step:**  
Iterate through more complex example parameter inputs.

## 2026-06-07 - MLflow tracking and longer conditional DDPM run

Question:
Does a longer conditional DDPM training run produce useful synthetic restoration outputs, and does 100-step sampling improve over 50-step sampling?

Setup:
Training data: data/processed/synthetic_restoration/nosferatu_v0/.
Model: compact conditional DDPM U-Net.
Conditioning: noisy target concatenated with degraded input as two channels.
Training objective: epsilon/noise prediction.
Sampling variants: 50-step DDPM sampling and 100-step DDPM sampling.
Evaluation: synthetic restoration metrics grouped by degradation type, using the same evaluation harness as the classical baselines.

Result:
The 50-step model produces visually promising restorations in some examples, especially where synthetic blemishes are visibly removed. However, quantitative metrics remain weak compared with the classical baselines and compared with the degraded input. The model appears to change too much of the image, leading to low PSNR/SSIM despite visually plausible blemish removal. The 100-step sampler improves some metrics, particularly for scratch_dust, but is not clearly or consistently better than 50-step sampling.

What worked:
The end-to-end diffusion pipeline now works: training, checkpoint loading, sampling, prediction manifest creation, metric evaluation, and visual-grid generation. Training loss decreased substantially during the CPU run, suggesting the model is learning the denoising objective. The model shows early qualitative evidence of restoration behaviour, especially for local blemish removal, and is visually restoring the origin.

What failed / looked suspicious:
The diffusion outputs do not yet beat the baselines. Full-image fidelity metrics are worse than the degraded input across degradation types, suggesting over-editing or poor conditioning. High-frequency ratios remain below 1, indicating possible over-smoothing and loss of archival texture. Increasing sampling from 50 to 100 steps provides only modest benefit and does not solve the main fidelity problem.

Next step:
Introduce explicit train/validation/test splits grouped by source frame, so evaluation is cleaner and leakage is avoided. Then run a better-controlled training experiment using training examples only, validation for monitoring/checkpoint selection, and a held-out test set for final comparison. Also tidy the repository by ignoring generated model/prediction folders, keeping only selected metrics and documentation figures under version control.

## 2026-06-21 - Blinded human review of DDPM restoration outputs

Question:
Are the current automatic restoration metrics too critical of visually plausible diffusion outputs, and do blinded human ratings agree with the metric-based ranking of restoration methods?

Setup:
A browser-based blinded human review workflow was created for held-out test-split examples. Each review item showed the degraded input, the synthetic ground-truth target, and one anonymous model output. The reviewer did not see the method name during scoring. Outputs were rated on artifact removal, detail preservation, texture authenticity, over-smoothing, hallucination risk, overall restoration quality, and whether the output would be chosen for restoration. The review covered 21 test examples across three degradation types, with seven anonymous outputs per example: classical baselines, DDPM 50-step sampling, DDPM 100-step sampling, and a test-split DDPM 50-step sample.

Result:
The blinded review showed that the DDPM outputs were preferred substantially more often than the automatic metrics suggested. conditional_ddpm_v1_50_steps and conditional_ddpm_v1_100_steps achieved the highest mean overall human scores, approximately 3.24 and 3.19 respectively. The 100-step DDPM had the highest yes/maybe restoration-choice rate, while the 50-step DDPM was effectively tied on overall quality. Classical median and non-local-means baselines were rated poorly despite stronger pixel-fidelity metrics in earlier automatic evaluation. The mask-informed inpaint_or_denoise baseline remained strongest for scratch_dust, where access to the synthetic damage mask gives it an advantage. Given the loss curves, the split-aware DDPM training does not show clear evidence of classical overfitting. Training loss and validation denoising loss both fall quickly and remain broadly aligned, with no sustained rise in validation loss or widening train/validation gap. This suggests that the weaker human-review performance of the split-aware/test-split model is unlikely to be explained by overfitting alone. A more plausible interpretation is objective mismatch: the diffusion denoising loss is being optimised successfully, but it is not sufficiently aligned with the desired restoration qualities captured in the human rubric, such as artifact removal, texture authenticity, limited over-editing, and preservation of archival detail.

What worked:
The blinded browser-based review workflow worked end-to-end and produced a CSV of human ratings without requiring spreadsheet software. The review was conducted on test-split examples, making it a stronger evaluation workflow than the earlier full-dataset visual checks. The results indicate that the diffusion model is producing visually meaningful restoration outputs that are not fully captured by PSNR, SSIM, or MAE. The human review also revealed degradation-specific behaviour: diffusion was preferred for blur/contrast and grain examples, while mask-informed inpainting remained strongest for local scratch/dust artifacts.

What failed / looked suspicious:
Automatic metrics appear overly punitive for diffusion outputs that are visually plausible but not pixel-aligned with the synthetic target. PSNR/SSIM may penalise acceptable perceptual changes, while median and denoising baselines may be rewarded for conservative or smoothing behaviour that is not preferred by the reviewer. 50-step test-split DDPM is clearly overfitting, a result corroborated by the test-loss curve.

Next step:
As the split-aware model showed weaker human evaluation, the next experiments should therefore focus less on early stopping and more on objective and sampling changes: residual prediction, x0 prediction, stronger conditioning, larger/more diverse training data, and evaluation using both automatic metrics and blinded human scores. Following this, human ratings can be used as a calibration layer alongside automatic metrics. Future evaluations should report both fidelity metrics and human-calibrated restoration criteria. The next model iteration should train on the train split only, monitor validation loss, evaluate on the held-out test split, and compare outputs using both automatic summaries and blinded human review.
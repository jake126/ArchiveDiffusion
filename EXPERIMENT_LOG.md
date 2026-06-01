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

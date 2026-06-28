# ArchiveDiffusion: Authenticity-Preserving Restoration of Public-Domain Film Stills

## Abstract

This project investigates diffusion-based restoration for degraded public-domain film stills. The goal is to improve perceptual quality by reducing grain, blur, scratches, and compression artefacts while preserving the archival texture and cinematic identity of the original material. The project begins with a compact DDPM-style baseline and extends toward paired restoration using synthetic degradation, quantitative metrics, qualitative analysis, and sampling-speed experiments.

## 1. Motivation

Archival film stills often contain visual degradation introduced by ageing, digitisation, compression, or damage to source material. Generative restoration methods can improve visual clarity, but may also hallucinate details or remove historically meaningful texture. This project treats restoration as a domain-specific generative modelling problem rather than a generic image enhancement task.

## 2. Research question

Can diffusion-based restoration improve degraded archival film stills while preserving their period-specific visual character?

## 3. Background

This section will summarise DDPMs, forward noising, reverse denoising, U-Net noise prediction, conditional restoration, and scheduler-based sampling.

## 4. Dataset

The first ArchiveDiffusion pilot uses the opening 20 minutes of *Nosferatu* (1922), sourced from the Internet Archive public-domain item `Nosferatu_DVD_quality`. Frames were extracted at one frame every two seconds, converted to grayscale, resized to width 512 pixels, and filtered using OCR-assisted text detection to remove intertitles and title cards.


## 5. Synthetic degradation

This section defines the degradation pipeline used to create paired training examples, including grain, blur, scratches, contrast loss, compression, and missing patches.

The non-text frames are divided into two candidate sets. The first contains relatively clean archive frames with lower visible degradation. These frames are used to learn a domain-specific visual prior and to create synthetic restoration pairs. The second contains naturally degraded archive frames with stronger visible grain, scratches, marks, uneven exposure, or other artefacts. These are reserved as a real archival restoration stress test.

![ArchiveDiffusion evaluation design](../outputs/figures/archive_diffusion_three_use_cases.png)

The figure above shows the two evaluation regimes. In the synthetic setting, artificial scratches, dust, and grain are added to a clean-ish archive frame. The original frame is retained as a known proxy target, enabling fidelity metrics such as PSNR, SSIM, and edge similarity. In the natural archival setting, the model is applied to genuinely degraded frames where no ground-truth restoration exists. These outputs will be evaluated using structured qualitative review focused on blemish reduction, detail preservation, hallucination avoidance, and preservation of archival film texture.

The central goal is not to erase all grain. ArchiveDiffusion treats restoration as an authenticity-preserving generative problem: improving legibility and reducing unwanted degradation while preserving the visual identity of silent-era film.

## 6. Model

This section will describe the model architecture, scheduler, training objective, resolution, hyperparameters, and implementation stack.

## 7. Experiments

### 7.1 Baseline

Baseline evaluation shows that simple classical methods behave differently across degradation types. Median filtering is effective for synthetic grain, mask-based inpainting is strong for local scratch/dust artefacts, and none of the simple baselines substantially improves blur/contrast degradation. These results justify reporting metrics by degradation type and motivate diffusion models as a flexible restoration approach, particularly where the restoration requires both local artefact removal and preservation or reconstruction of archival texture.



### 7.2 Unconditional DDPM baseline

To be completed.

### 7.3 Restoration model

To be completed.

### 7.4 Authenticity preservation analysis

To be completed.

### 7.5 Sampling acceleration

To be completed.

## 8. Evaluation

Planned metrics include PSNR, SSIM, optional LPIPS, edge preservation, texture retention, and qualitative side-by-side comparison.

The high-frequency ratio is used as an initial over-smoothing diagnostic. Values far below 1 indicate loss of texture relative to the target, while values far above 1 indicate excess noise or artefacts. This metric is interpreted alongside PSNR/SSIM because restoration quality is not equivalent to smoothing.


### Blinded human review

The initial automatic evaluation suggested that the conditional DDPM outputs were weak relative to classical baselines, particularly under PSNR, SSIM, and MAE. However, visual inspection suggested that the diffusion model was often removing visible blemishes and producing more plausible restorations than these metrics implied. To test whether the evaluation criteria were too pixel-fidelity-driven, I added a blinded human review workflow.

For each held-out test example, the reviewer was shown the degraded input, the synthetic ground-truth target, and one anonymous model output. Method names were hidden during scoring. Each output was rated for artifact removal, detail preservation, texture authenticity, over-smoothing, hallucination risk, overall restoration quality, and whether the reviewer would choose it as a restoration. I tested a 50- and 100-step DDPM trained before implementing the test-train split ("pre-split"), and a 50-step DDPM split-aware model ("post-split").

The review covered 21 held-out test examples across blur/contrast, grain, and scratch/dust degradation types, with seven anonymous outputs per example. The results showed that the DDPM outputs were preferred more often than the automatic metrics suggested. The pre-split 50-step and 100-step DDPM samplers achieved the highest mean overall human scores, with the 100-step sampler receiving the highest yes/maybe restoration-choice rate. Classical median and non-local-means baselines were rated poorly, even though they can perform well on pixel-level metrics. This suggests that PSNR and SSIM alone over-reward conservative smoothing and under-value perceptually useful restoration. The post-split achieved inferior result, indicating clear overfitting in the pre-split cases, but still comfortably outperformed the baseline models (average score of 2.7 across the metrics compared to the best baseline of 2.4 for inpainting).

The results were degradation-specific. Diffusion outputs were preferred for blur/contrast and grain examples, where classical filters often produced over-smoothed or visually unconvincing results. For scratch/dust examples, the mask-informed inpainting baseline remained strongest, which is expected because it has access to the synthetic damage mask. This highlights the need to report restoration performance by degradation type and to distinguish mask-informed baselines from methods that operate without oracle damage information.

These findings motivate a two-layer evaluation protocol. Automatic metrics remain useful for detecting over-editing, loss of texture, and structural drift, but they should be interpreted alongside blinded human ratings that assess restoration quality, archival texture, and visual plausibility.


## 9. Results

### Model calibration review: conditional DDPM vs residual DDPM

After the initial blinded review showed that diffusion outputs were preferred over classical baselines more often than pixel-level metrics suggested, I ran a harder model-calibration review focused only on diffusion variants. This review compared five outputs per held-out test example: the previous split-aware 50-step DDPM reference, the previous non-split-aware 100-step DDPM reference, a longer split-aware conditional DDPM, residual DDPM with full correction, and residual DDPM with conservative correction.

Residual DDPM was introduced to address a weakness identified in the earlier review: full-image conditional DDPM can remove visible degradation, but may also over-edit the image, reduce archival texture, or drift away from the input. Instead of generating a whole restored frame, residual DDPM predicts a correction to the degraded input. The restored image is then formed by adding the predicted correction back to the input. This biases the model toward preservation rather than full-frame regeneration. Two residual variants were evaluated: full correction and conservative correction.

The second blinded review covered 21 held-out test examples and 105 anonymous model outputs. The previous DDPM models remained strongest. `conditional_ddpm_v1_100_steps` achieved the highest mean overall human score, 3.19 / 5, while `conditional_ddpm_v1_50_steps` was effectively tied at 3.14 / 5. Both had a yes/maybe restoration-choice rate of 66.7%. This confirms that the earlier preference for DDPM was not simply caused by comparison against weak classical baselines: the earlier DDPM outputs remained competitive in a harder diffusion-only comparison.

The new variants were informative but did not outperform the earlier DDPM references. The residual DDPM full-correction model was the strongest new variant, with a mean overall score of 2.76 and a yes/maybe rate of 52.4%. The longer split-aware conditional DDPM scored 2.62 overall with a yes/maybe rate of 38.1%. The conservative residual model scored lowest overall, 2.38, although it had the lowest over-smoothing and hallucination-risk ratings. This suggests that conservative residual correction did reduce some risks associated with over-editing, but at the cost of insufficient restoration strength.

These findings suggest that the main limitation is unlikely to be sampling length alone. The earlier 50-step and 100-step DDPM outputs remained very close in human evaluation, while the longer split-aware run did not close the gap to the older non-split-aware models. A plausible explanation is reduced data exposure: the split-aware model sees fewer training frames than the original full-data model. The next priority is therefore to expand the training dataset while preserving split-aware evaluation, for example by using more clean-ish source frames and multiple synthetic degradation variants per frame. Residual correction remains a promising restoration-specific idea, but it likely needs either more training data or a tuned correction-strength sweep to balance artifact removal against archival preservation.


## 10. Failure cases

To be completed.

## 11. Responsible use and limitations

Restored outputs should be interpreted as plausible enhancements, not historically verified reconstructions. The model may hallucinate detail, over-smooth texture, or introduce artefacts. The project will use public-domain or permissively licensed materials only.

## 12. Next steps

To be completed.


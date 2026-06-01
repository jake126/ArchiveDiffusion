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

## 9. Results

To be completed.

## 10. Failure cases

To be completed.

## 11. Responsible use and limitations

Restored outputs should be interpreted as plausible enhancements, not historically verified reconstructions. The model may hallucinate detail, over-smooth texture, or introduce artefacts. The project will use public-domain or permissively licensed materials only.

## 12. Next steps

To be completed.


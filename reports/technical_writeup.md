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

This section will describe the public-domain film sources, frame extraction process, pre-processing, split strategy, and licensing notes.

## 5. Synthetic degradation

This section will define the degradation pipeline used to create paired training examples, including grain, blur, scratches, contrast loss, compression, and missing patches.

## 6. Model

This section will describe the model architecture, scheduler, training objective, resolution, hyperparameters, and implementation stack.

## 7. Experiments

### 7.1 Unconditional DDPM baseline

To be completed.

### 7.2 Restoration model

To be completed.

### 7.3 Authenticity preservation analysis

To be completed.

### 7.4 Sampling acceleration

To be completed.

## 8. Evaluation

Planned metrics include PSNR, SSIM, optional LPIPS, edge preservation, texture retention, and qualitative side-by-side comparison.

## 9. Results

To be completed.

## 10. Failure cases

To be completed.

## 11. Responsible use and limitations

Restored outputs should be interpreted as plausible enhancements, not historically verified reconstructions. The model may hallucinate detail, over-smooth texture, or introduce artefacts. The project will use public-domain or permissively licensed materials only.

## 12. Next steps

To be completed.


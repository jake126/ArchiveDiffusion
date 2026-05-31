# Archive Diffusion

**Authenticity-preserving diffusion restoration for public-domain film stills.**

ArchiveDiffusion is a small-scale research and engineering project combining my love of old film with curiosity of new computer vision techniques. I will explore how diffusion-based image restoration can improve degraded archival film stills while preserving their original cinematic character - we don't want to yassify Boris Karloff, just loving restore the frames. The project is designed as a local and reproducible modelling study, with a clear path from foundational DDPM training to practical restoration, evaluation, and accelerated sampling experiments.

## Motivation

Inspired by a readthrough of Mark Cousins' "The Story of Film", many archival film stills contain grain, scratches, blur, low contrast, and other forms of degradation. Modern generative image models can enhance or hallucinate detail, but restoration can easily become over-smoothed, ahistorical, or aesthetically inauthentic.

This project asks: can diffusion-based restoration improve archival film still quality while preserving the period-specific texture and visual identity of the source material? And is this possible on a 6-year-old i5 HP EliteBook combined with a general reticence to pay for a single penny of compute?

## Initial research goals

1. Build a reproducible public-domain film-frame dataset pipeline.
2. Train a compact Denoising Diffusion Probabilistic Models (DDPM)-style diffusion model on film stills to understand the mechanics of diffusion from the ground up.
3. Develop a synthetic degradation pipeline for grain, blur, scratches, compression, contrast loss, and missing regions.
4. Train and evaluate restoration models that map degraded stills to cleaner versions.
5. Study the trade-off between perceptual enhancement and authenticity preservation.
6. Explore reduced-step sampling or scheduler comparison as a lightweight acceleration experiment.

## Why this project

The project is intended to demonstrate:

* practical experience with diffusion models;
* deep learning implementation in PyTorch / Diffusers;
* domain-specific generative modelling;
* media restoration and evaluation;
* thoughtful handling of hallucination, authenticity, and model behaviour;
* clear research communication through a public technical write-up.

## Planned outputs

* Dataset preparation scripts for extracting and pre-processing public-domain film frames
* Synthetic degradation functions
* A compact unconditional DDPM baseline
* A conditional restoration model or restoration pipeline
* Quantitative metrics: PSNR, SSIM, optionally LPIPS/FID/KID
* Meaningful evaluation metrics (derivation and tracking)
* Qualitative before/after restoration grids
* Sampling-speed versus output-quality comparison
* Final technical report

## Evaluation layer

The repository contains evaluations to track quantitative fidelity, perceptual/archival quality, robustness across degradation types, sampling-speed trade-offs, and a small nearest-neighbour memorisation audit.

## Repository structure

```text
ArchiveDiffusion/
  README.md
  PROJECT\_SPEC.md
  ROADMAP.md
  EXPERIMENT\_LOG.md
  ENVIRONMENT.md
  configs/
    baseline\_ddpm.yaml
  docs/
    dataset\_notes.md
    research\_questions.md
  notebooks/
    01\_dataset\_exploration.ipynb
    02\_train\_baseline\_ddpm.ipynb
    03\_restoration\_demo.ipynb
    04\_evaluation.ipynb
  src/
    archivediffusion/
      degradation.py
      dataset.py
      metrics.py
      sample.py
      train.py
  outputs/
    figures/
    samples/
  reports/
    technical\_writeup.md
```

## Status

Version 0.1: project specification and repository scaffold.


30/05/2026
The first dataset pilot uses the opening 20 minutes of \*Nosferatu\* (1922), sourced from the Internet Archive public-domain item `Nosferatu\_DVD\_quality`. This pilot corpus is used to validate the download, frame extraction, grain-ranking, and manual-curation workflow before scaling to additional public-domain films.

Raw videos and extracted frames are not committed to the repository. Dataset provenance and processing steps are documented in `docs/dataset\_notes.md` and `configs/dataset\_manifest.csv`. The main step taken here was using an OCR filter to remove text-based intertitle frames from analysis, removing 168 frames.


31/05/2026

The initial modelling strategy separates the non-text frames into two groups:

1. Clean-ish archive frames: relatively stable frames with lower visible degradation. These are used to learn a domain-specific archival visual prior and to create synthetic restoration pairs.
2. Naturally degraded archive frames: frames with stronger visible grain, scratches, marks, uneven exposure, or other artefacts. These are reserved as a real-world restoration stress test.

The evaluation design uses both controlled and natural degradation. Synthetic degradations applied to clean-ish frames provide known targets for fidelity metrics such as PSNR and SSIM. Naturally degraded frames are evaluated using a structured qualitative rubric focused on blemish removal, detail preservation, hallucination avoidance, and preservation of archival film texture.

# Archive Diffusion

**Authenticity-preserving diffusion restoration for public-domain film stills.**

ArchiveDiffusion is a small-scale research and engineering project combining my love of old film with curiosity of new computer vision techniques. I will explore how diffusion-based image restoration can improve degraded archival film stills while preserving their original cinematic character - we don't want to yassify Humphrey Bogart, just loving restore the frames. The project is designed as a local and reproducible modelling study, with a clear path from foundational DDPM training to practical restoration, evaluation, and accelerated sampling experiments.

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

- practical experience with diffusion models;
- deep learning implementation in PyTorch / Diffusers;
- domain-specific generative modelling;
- media restoration and evaluation;
- thoughtful handling of hallucination, authenticity, and model behaviour;
- clear research communication through a public technical write-up.

## Planned outputs

- Dataset preparation scripts for extracting and pre-processing public-domain film frames
- Synthetic degradation functions
- A compact unconditional DDPM baseline
- A conditional restoration model or restoration pipeline
- Quantitative metrics: PSNR, SSIM, optionally LPIPS/FID/KID
- Meaningful evaluation metrics (derivation and tracking)
- Qualitative before/after restoration grids
- Sampling-speed versus output-quality comparison
- Final technical report

## Evaluation layer

The repository contains evaluations to track quantitative fidelity, perceptual/archival quality, robustness across degradation types, sampling-speed trade-offs, and a small nearest-neighbour memorisation audit.

## Repository structure

```text
ArchiveDiffusion/
  README.md
  PROJECT_SPEC.md
  ROADMAP.md
  EXPERIMENT_LOG.md
  ENVIRONMENT.md
  configs/
    baseline_ddpm.yaml
  docs/
    dataset_notes.md
    research_questions.md
  notebooks/
    01_dataset_exploration.ipynb
    02_train_baseline_ddpm.ipynb
    03_restoration_demo.ipynb
    04_evaluation.ipynb
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
    technical_writeup.md
```

## Status

Version 0.1: project specification and repository scaffold.


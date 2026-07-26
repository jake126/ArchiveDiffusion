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

Note that a command log and all scripts are included in this repo, so results should be fully reproducible; if there are any issues getting things working, please get in touch! Note also that a good chunk of this report and the underlying code have been built with my research assistant ChatGPT (5.4 and 5.5). The important stuff is handwritten, but apologies for any legacy "here's the code you asked for" comments and repo artifacts.

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

The repository contains evaluations to track quantitative fidelity, perceptual/archival quality, robustness across degradation types, sampling-speed trade-offs, and a human review!

## Repository structure

```text
ArchiveDiffusion/
  README.md
  PROJECT\_SPEC.md
  ROADMAP.md
  EXPERIMENT\_LOG.md
  configs/
    baseline\_ddpm.yaml
    dataset\_manifest.csv
  docs/
    assets\
    dataset\_notes.md
    command\_log.md
    research\_questions.md
  scripts/
    many, many scripts for many, many tasks. 
  outputs/
    contact\_sheets
    evaluation\_grids
    figures/
    models/
    predictions/
    samples/
    training\_curves/
  reports/
    technical\_writeup.md
    assets/
```
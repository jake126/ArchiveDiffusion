# Project Specification: ArchiveDiffusion

## Working title

**ArchiveDiffusion: Authenticity-Preserving Restoration of Public-Domain Film Stills**

## One-sentence summary

ArchiveDiffusion uses diffusion-based image restoration to reduce grain, blur, scratches, and other degradation in public-domain film stills while preserving the cinematic texture and archival character of the source images.

## Core research question

Can diffusion-based restoration improve the perceptual quality of degraded archival film stills without over-smoothing, over-modernising, or hallucinating away historically meaningful visual texture?

## Secondary research questions

1. How does a compact DDPM-style model behave when trained from scratch on a narrow visual domain such as public-domain film stills?
2. Which synthetic degradation types are most useful for learning restoration: grain, blur, scratches, compression, contrast loss, or missing patches?
3. How do restoration metrics such as PSNR and SSIM relate to perceived authenticity and cinematic quality?
4. Can reduced-step sampling or scheduler changes provide a useful speed-quality trade-off for local restoration workflows?
5. Does the model reproduce memorised training examples, or does it generate/restoratively reconstruct broader domain structure?

## Scope

### In scope

- Public-domain film stills or legally usable frame extracts.
- Grayscale and/or RGB image restoration.
- Synthetic degradation and paired training data.
- Compact diffusion models runnable locally.
- Qualitative and quantitative evaluation.
- Lightweight acceleration experiments.
- Nearest-neighbour checks for memorisation risk.

### Out of scope for v0.1

- Full video diffusion.
- Commercial-scale training.
- Claims of historically accurate restoration.
- Training on copyrighted film material without permission.
- Recovering exact training images from third-party generative models.
- Real-time restoration.

## Target user / audience

The project is designed for a technical ML audience reviewing evidence of hands-on diffusion modelling, especially research engineering or applied research teams working on generative media models, scientific ML, or domain-specific generative systems.

## Dataset strategy

The initial dataset will be built from public-domain films or legally reusable stills. Frames will be extracted, filtered, resized, and cropped into a standard training resolution.

### Candidate data sources

- Public-domain films from archival collections.
- Internet Archive public-domain videos, subject to manual licence checks.
- Public-domain still images from film archives where reuse is clearly permitted.
- A small non-film starter dataset may be used only to validate the training loop.

### Initial preprocessing

- Extract frames at a low sampling rate, for example one frame every 2-5 seconds.
- Remove near-duplicate frames.
- Resize and centre/random crop to 64x64 for v0.1.
- Later extend to 128x128.
- Store metadata: source film, timestamp, licence note, preprocessing version.

## Synthetic degradation pipeline

The restoration task will use paired clean/degraded examples created through controlled synthetic degradation.

Candidate degradations:

1. Film grain / Gaussian noise.
2. Motion blur or Gaussian blur.
3. Contrast reduction.
4. JPEG compression artefacts.
5. Dust and scratches.
6. Small missing patches for inpainting-like restoration.
7. Downsampling and upsampling for low-resolution artefacts.

The initial degradation pipeline should be parameterised so that light, medium, and heavy degradation settings can be evaluated separately.

## Modelling plan

### Phase 1: foundational DDPM baseline

Train an unconditional diffusion model on film-frame crops.

Purpose:

- demonstrate understanding of forward noising and reverse denoising;
- establish a visual prior over film stills;
- produce sample grids from random noise;
- debug data, training, and sampling infrastructure.

Likely stack:

- Python;
- PyTorch;
- Hugging Face Diffusers;
- small U-Net architecture;
- DDPM scheduler.

### Phase 2: restoration model

Train or adapt a conditional restoration model using degraded/clean image pairs.

Candidate approaches:

1. Concatenate degraded image and noisy target as model input channels.
2. Use an image-to-image diffusion pipeline.
3. Use a pretrained restoration/inpainting model as a practical comparison baseline.

### Phase 3: authenticity-preserving restoration

Introduce evaluation and possibly loss/conditioning choices that reward preservation of film-like texture rather than maximum smoothing.

Candidate approaches:

- compare light, medium, and heavy restoration settings;
- measure high-frequency texture retention;
- compare edge preservation;
- include qualitative judgement grids;
- optionally conduct a small human preference study.

### Phase 4: acceleration / distillation-lite

Explore faster sampling methods and quality-speed trade-offs.

Candidate approaches:

- compare DDPM sampling against DDIM-style reduced-step sampling;
- test different inference step counts;
- report runtime and quality metrics;
- document trade-offs clearly.

## Evaluation plan

### Quantitative metrics

- PSNR for pixel-level reconstruction.
- SSIM for structural similarity.
- LPIPS if feasible for perceptual similarity.
- Optional FID/KID for sample distribution quality if sample size is sufficient.
- Edge preservation score.
- Texture/high-frequency retention measure.

### Qualitative outputs

- Before/after restoration grids.
- Degradation severity comparison.
- Side-by-side comparison with classical denoising/deblurring baselines.
- Failure case gallery.
# Evaluation Plan

ArchiveDiffusion is not only a visual restoration demo. The evaluation layer is designed to test whether diffusion-based restoration improves archival image quality while preserving the cinematic character of the source material.

## Evaluation Questions

1. Does the restoration model improve fidelity relative to synthetically degraded inputs?
2. Does the model preserve archival texture rather than over-smoothing or hallucinating detail?
3. How robust is the model across degradation types and severities?
4. What is the quality-speed trade-off under reduced-step sampling or alternative schedulers?
5. Are outputs overly similar to training frames, suggesting memorisation rather than restoration?

## Evaluation Tracks

### Track A: Reconstruction Fidelity

Used when a clean reference frame exists because degradation has been applied synthetically.

Metrics:

- PSNR
- SSIM
- LPIPS, optional
- MSE / MAE

Comparisons:

- degraded input vs clean target
- restored output vs clean target
- baseline classical filters vs diffusion restoration

### Track B: Perceptual and Archival Quality

Used for both synthetic and naturally degraded archival stills.

Criteria:

- perceived sharpness
- preservation of faces, objects, and edges
- preservation of film grain / texture
- absence of plastic smoothing
- absence of hallucinated modern artefacts
- consistency with the original cinematic mood

Outputs:

- side-by-side restoration grids
- reviewer notes
- optional small human preference study

### Track C: Robustness by Degradation Type

Stress tests the model across controlled degradation families.

Degradations:

- Gaussian noise / film grain
- blur
- compression
- scratches and dust
- contrast loss
- missing patches / inpainting masks

For each degradation family, evaluate light, medium, and heavy severity.

### Track D: Speed / Sampling Efficiency

Tests whether faster sampling sacrifices quality.

Comparisons:

- full-step DDPM sampling
- reduced-step DDPM sampling
- DDIM-style sampling, if implemented
- later: distilled or student model sampling, if implemented

Metrics:

- wall-clock inference time
- number of sampling steps
- PSNR / SSIM / LPIPS
- qualitative quality notes

### Track E: Memorisation / Nearest-Neighbour Audit

This is a small responsible-AI audit rather than the core project.

Question:

- Do generated or restored outputs closely reproduce training frames?

Methods:

- nearest-neighbour search in pixel space
- nearest-neighbour search in embedding space, optional
- visual inspection of closest training examples

The goal is not to extract training data, but to document whether the model is learning a general archival image prior or reproducing specific frames.

## Reporting Format

Each evaluation run should produce:

- config used
- dataset split
- degradation type and severity
- model checkpoint
- metric table
- restoration grid
- brief qualitative notes
- known failure modes

Use `evaluations/results/` for metric outputs and `evaluations/figures/` for generated figures.

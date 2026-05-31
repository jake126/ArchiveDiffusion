# Evaluation Plan

A diffusion restoration model trained on relatively clean archival frames with synthetic degradations can learn a domain-specific visual prior for *Nosferatu*-style imagery. This prior may improve restoration of naturally degraded frames, provided the model is evaluated not only for fidelity but also for preservation of archival texture.

## Evaluation Questions

1. Does the restoration model improve fidelity relative to synthetically degraded inputs?
2. Does the model preserve archival texture rather than over-smoothing or hallucinating detail?
3. How robust is the model across degradation types and severities?
4. What is the quality-speed trade-off under reduced-step sampling or alternative schedulers?
5. Are outputs overly similar to training frames, suggesting memorisation rather than restoration?

# Evaluation Plan

ArchiveDiffusion evaluates restoration quality across two complementary settings: controlled synthetic degradation and naturally degraded archival material.

## Evaluation setting A: synthetic degradation fidelity

In this setting, relatively clean archive frames are treated as proxy targets. Synthetic degradations such as scratches, dust marks, blur, contrast loss, compression artefacts, or additional grain are applied to create paired examples.

The model input is the synthetically degraded frame. The target is the original clean-ish archive frame.

This setting supports quantitative fidelity evaluation because the target is known. Initial metrics will include PSNR, SSIM, and edge similarity, with LPIPS considered as a later perceptual metric. These metrics are interpreted as reconstruction-fidelity measures rather than complete measures of restoration quality.

## Evaluation setting B: naturally degraded archival restoration

In this setting, the model is applied to naturally degraded frames selected from the source film. These frames contain real archival artefacts such as scratches, marks, heavy grain, blotches, low-light damage, compression artefacts, or uneven exposure.

No ground-truth restored target exists for these frames. Evaluation therefore uses structured qualitative review. Outputs are assessed for blemish reduction, preservation of meaningful visual detail, introduction of new artefacts, and overall restoration usefulness.

## Evaluation setting C: authenticity and over-smoothing

ArchiveDiffusion does not aim to erase all film grain. The goal is authenticity-preserving restoration: improving legibility and reducing unwanted degradation while preserving the visual texture and cinematic identity of archival footage.

For both synthetic and naturally degraded examples, outputs will be reviewed for grain preservation, contrast preservation, plausible detail treatment, hallucination risk, and over-smoothing. A successful restoration should not make a 1922 film frame look artificially modern, waxy, or ahistorical.


## Evaluation setting D: Speed / Sampling Efficiency

Tests whether faster sampling sacrifices quality.

Comparisons:

- full-step DDPM sampling
- reduced-step DDPM sampling

Metrics:

- wall-clock inference time
- number of sampling steps
- PSNR / SSIM / LPIPS
- qualitative quality notes

### Evaluation setting E: Memorisation / Nearest-Neighbour Audit

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

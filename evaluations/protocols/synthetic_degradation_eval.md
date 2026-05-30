# Protocol: Synthetic Degradation Evaluation

## Purpose

Evaluate restoration quality when clean targets are available.

## Dataset

1. Select held-out clean frames from the validation split.
2. Apply controlled synthetic degradation.
3. Restore degraded frames using the trained model.
4. Compare restored outputs with clean targets.

## Degradation Matrix

| Family | Light | Medium | Heavy |
|---|---:|---:|---:|
| Gaussian noise | sigma 0.05 | sigma 0.10 | sigma 0.20 |
| Blur | kernel 3 | kernel 5 | kernel 9 |
| JPEG compression | quality 70 | quality 40 | quality 20 |
| Scratches / dust | sparse | moderate | dense |
| Contrast loss | mild | moderate | severe |
| Missing patches | 5% | 15% | 30% |

## Metrics

- PSNR
- SSIM
- LPIPS, optional
- runtime per image

## Outputs

- `evaluations/results/synthetic_degradation_metrics.csv`
- `evaluations/figures/synthetic_degradation_grid.png`
- notes in `EXPERIMENT_LOG.md`

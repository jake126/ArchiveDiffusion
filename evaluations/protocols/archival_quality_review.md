# Protocol: Archival Quality Review

## Purpose

Assess whether restoration improves image quality while preserving archival character.

## Review Criteria

Score each output from 1 to 5.

| Criterion | Question |
|---|---|
| Clarity | Is the restored image easier to interpret? |
| Texture preservation | Does it retain plausible film grain / archival texture? |
| Over-smoothing | Does it look plasticky or artificially polished? |
| Hallucination control | Are new details plausible and not misleading? |
| Cinematic identity | Does the image still feel like the original film source? |
| Overall preference | Which image would be preferable for archival presentation? |

## Review Setup

Use side-by-side grids:

1. original degraded frame
2. restored output
3. clean target, if available
4. classical baseline, if available

## Outputs

- `evaluations/human_review/review_template.csv`
- `evaluations/figures/archival_review_grid.png`

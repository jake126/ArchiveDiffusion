# Roadmap

## Milestone 0: repo and research framing

- [x] Create repository scaffold.
- [x] Draft project specification.
- [x] Draft technical write-up skeleton.
- [x] Draft CV/application bullets.
- [ ] Choose final project name.
- [ ] Choose initial public-domain source films.

## Milestone 1: dataset pipeline

- [ ] Identify 3-5 public-domain films or image sources.
- [ ] Record licence/source notes.
- [ ] Extract frames at fixed intervals.
- [ ] Remove near-duplicates.
- [ ] Resize/crop to 64x64.
- [ ] Create train/validation/test split.
- [ ] Save sample contact sheet.

## Milestone 2: degradation pipeline

- [ ] Implement grain/noise degradation.
- [ ] Implement blur degradation.
- [ ] Implement contrast loss.
- [ ] Implement scratches/dust overlay.
- [ ] Implement missing patch degradation.
- [ ] Add light/medium/heavy degradation configs.
- [ ] Save before/after degradation grid.

## Milestone 3: unconditional DDPM baseline

- [ ] Implement dataset loader.
- [ ] Configure compact U-Net.
- [ ] Train at 64x64.
- [ ] Save loss curves.
- [ ] Generate random sample grid.
- [ ] Document model architecture and training settings.

## Milestone 4: restoration experiment

- [ ] Create paired degraded/clean dataset.
- [ ] Train initial restoration model on one degradation type.
- [ ] Evaluate PSNR and SSIM.
- [ ] Produce before/after restoration grid.
- [ ] Document first failure cases.

## Milestone 5: authenticity evaluation

- [ ] Compare light/medium/heavy restoration.
- [ ] Add edge-preservation metric.
- [ ] Add texture-retention or high-frequency metric.
- [ ] Add qualitative discussion of over-smoothing and hallucination.

## Milestone 6: acceleration experiment

- [ ] Compare sampling schedulers or step counts.
- [ ] Record runtime versus quality metrics.
- [ ] Add speed-quality table to report.

## Milestone 7: application-ready polish

- [ ] Finalise README.
- [ ] Finalise technical report.
- [ ] Add selected visual outputs.
- [ ] Add reproducibility instructions.
- [ ] Add concise CV bullets.
- [ ] Link repository in application.


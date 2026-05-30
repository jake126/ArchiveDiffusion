# Environment

Initial suggested environment. This may be updated once implementation begins.

## Python

Python 3.10 or 3.11.

## Core packages

```bash
pip install torch torchvision diffusers accelerate transformers datasets pillow opencv-python scikit-image numpy pandas matplotlib tqdm pyyaml
```

Optional perceptual metrics:

```bash
pip install lpips clean-fid
```

Optional development tools:

```bash
pip install black ruff pytest jupyter
```

## Hardware assumptions

* First tests will run on CPU with a tiny dataset and tiny model.
* Useful training likely requires a local GPU or cloud notebook.
* Sticking to 64x64 crops as we're more interested in getting things running than optimising for quality.

## Reproducibility notes

I will endeavour to:

* Record dataset source and licence for every film.
* Save train/validation/test split files.
* Save config files for each experiment.
* Save random seed where relevant.


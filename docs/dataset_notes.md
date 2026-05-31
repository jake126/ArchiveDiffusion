# Dataset Notes

# Principles

* Use only public-domain or clearly permissively licensed sources.
* Keep a record of every source film, source URL, licence note, and extraction settings.
* Do not assume that all archival-looking footage is public domain.
* Prefer a small, clean, well-documented dataset over a large ambiguous one.



\## Nosferatu pilot corpus



The first pilot dataset uses the opening section (20 minutes) of \*Nosferatu\* (1922), sourced from the Internet Archive item.



Source item: Internet Archive, `Nosferatu\_DVD\_quality`

Source link: https://archive.org/download/Nosferatu\_DVD\_quality

Selected file: `nosferatu-1of5\_512kb.mp4` for initial pipeline development; `nosferatu-1of5.mpg` reserved as a higher-quality source for later experiments.

Usage status: listed by Internet Archive as Public Domain. The item page notes that the film was released in 1922 and is no longer eligible for copyright.

Initial segment: first \~20 minutes of part 1.

Initial sampling plan: 0.5 frame per second, grayscale, resized to width 512 pixels.

Raw frame count: 595.

Rationale: \*Nosferatu\* provides high-contrast silent-era cinematography, visible film texture, grain, compression artefacts, scratches, intertitles, and low-light scenes. This makes it a useful pilot source for testing authenticity-preserving restoration methods before expanding to a broader public-domain film corpus.

Notes: 

* The first 20 minutes contain a high proportion of title and intertitle cards. To avoid an overfitting on text restoration, I implemented an EasyOCR filter to remove frames with a high likelihood of text content - this removed 168 frames.

Important caveat: the first pilot dataset is intentionally narrow and should be treated as a pipeline validation corpus, not a general-purpose archival restoration dataset. I will add more fun films for this purpose at a later date.




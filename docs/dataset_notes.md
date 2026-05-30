# Dataset Notes

## Principles

- Use only public-domain or clearly permissively licensed sources.
- Keep a record of every source film, source URL, licence note, and extraction settings.
- Do not assume that all archival-looking footage is public domain.
- Prefer a small, clean, well-documented dataset over a large ambiguous one.

## Candidate selection criteria

- Public-domain status is clear.
- Visual style is relevant: grain, contrast variation, scratches, archival character.
- Sufficient resolution for frame extraction.
- No unnecessary personal or sensitive content.
- Domain is coherent enough for a compact model to learn.

## Metadata fields to track

| Field | Description |
|---|---|
| source_id | Short unique source identifier |
| title | Film or collection title |
| source_url | Link to source page |
| licence_note | Public-domain or licence statement |
| extraction_rate | Frame sampling interval |
| frame_count | Number of extracted frames |
| preprocessing_version | Resize/crop/filter pipeline version |
| notes | Any caveats |


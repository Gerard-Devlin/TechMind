"""Expose local course artwork to Ask AI without a second course registry."""

from pathlib import Path
import re


def on_config(config):
    docs_dir = Path(config.docs_dir)
    artwork_dir = docs_dir / "assets" / "start" / "courses"
    artwork = {}
    for path in sorted(artwork_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg", ".avif"}:
            slug = re.sub(r"[^a-z0-9]", "", path.stem.lower())
            if slug:
                artwork[slug] = path.relative_to(docs_dir).as_posix()
    config.extra["ask_ai_course_images"] = artwork
    return config

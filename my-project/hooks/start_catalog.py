"""Derive Start-page course metadata from the configured navigation."""

from datetime import datetime
from pathlib import Path
import subprocess


def _markdown_paths(node):
    if isinstance(node, str):
        if node.lower().endswith(".md"):
            yield node
        return

    if isinstance(node, list):
        for child in node:
            yield from _markdown_paths(child)
        return

    if isinstance(node, dict):
        for child in node.values():
            yield from _markdown_paths(child)


def _git_date(project_dir, docs_dir, paths):
    existing = [(docs_dir / path).resolve() for path in paths if (docs_dir / path).is_file()]
    if not existing:
        return None

    try:
        repo_root = Path(
            subprocess.run(
                ["git", "-C", str(project_dir), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        pathspecs = [str(path.relative_to(repo_root)) for path in existing]
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%ct", "--", *pathspecs],
            check=False,
            capture_output=True,
            text=True,
        )
        timestamp = result.stdout.strip()
        if timestamp.isdigit():
            return datetime.fromtimestamp(int(timestamp))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    # New, untracked courses still get a useful date before their first commit.
    return datetime.fromtimestamp(max(path.stat().st_mtime for path in existing))


def on_config(config):
    """Expose generated short names and last-modified dates for top-level courses."""

    project_dir = Path(config.config_file_path).resolve().parent
    docs_dir = Path(config.docs_dir).resolve()
    course_dates = {}
    course_short_names = {}

    for entry in config.nav or []:
        if not isinstance(entry, dict):
            continue
        for title, children in entry.items():
            if not isinstance(children, (list, dict)):
                continue
            paths = list(_markdown_paths(children))
            if paths:
                course_short_names[title] = Path(paths[0]).parts[0]
            modified = _git_date(project_dir, docs_dir, paths)
            if modified:
                course_dates[title] = modified.strftime("%b %d, %Y").replace(" 0", " ")

    config.extra["start_course_dates"] = course_dates
    config.extra["start_course_short_names"] = course_short_names
    return config

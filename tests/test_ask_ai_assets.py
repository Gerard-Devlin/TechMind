"""Check generated artwork metadata and safe HTML attribute serialization."""

import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from jinja2 import Environment, FileSystemLoader


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ask_ai_assets", ROOT / "my-project/hooks/ask_ai_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class RootAttributes(HTMLParser):
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "data-tm-ai-root" in values:
            self.values = values


class AskAIAssetsTests(unittest.TestCase):
    def test_inventory_is_generated_from_existing_local_files(self):
        config = SimpleNamespace(docs_dir=str(ROOT / "my-project/docs"), extra={})
        assets.on_config(config)
        images = config.extra["ask_ai_course_images"]
        self.assertEqual(images["coa"], "assets/start/courses/coa.webp")
        self.assertEqual(images["mldl"], "assets/start/courses/mldl.webp")
        self.assertTrue(all((Path(config.docs_dir) / path).is_file() for path in images.values()))

    def test_partial_serializes_inventory_without_breaking_attributes(self):
        inventory = {"coa": "assets/start/courses/coa.webp", "mldl": "assets/start/courses/mldl.webp"}
        config = SimpleNamespace(extra={"ask_ai_course_images": inventory}, site_url="https://gerard-devlin.github.io/TechMind/")
        env = Environment(loader=FileSystemLoader(ROOT / "my-project/material"), autoescape=True)
        html = env.get_template("partials/ask-ai.html").render(config=config)
        parser = RootAttributes()
        parser.feed(html)
        self.assertEqual(json.loads(parser.values["data-course-images"]), inventory)


if __name__ == "__main__":
    unittest.main()

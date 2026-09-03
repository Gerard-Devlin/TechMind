"""Run with: python -m unittest discover -s my-project/tests -v"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import markdown
from markdown.extensions.toc import TocExtension
from mkdocs.structure.toc import get_toc


HOOK_PATH = Path(__file__).resolve().parents[1] / "hooks" / "toc_inline_code.py"
SPEC = importlib.util.spec_from_file_location("toc_inline_code", HOOK_PATH)
HOOK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOK)


def render(source, fixed=True, **options):
    toc = HOOK.InlineCodeTocExtension if fixed else TocExtension
    md = markdown.Markdown(
        extensions=[toc(**options), "attr_list", "pymdownx.inlinehilite"]
    )
    return md, md.convert(source)


class InlineCodeTocTests(unittest.TestCase):
    def test_inline_and_sidebar_toc_preserve_multiple_code_spans(self):
        md, output = render("[TOC]\n\n## 一、`repr()`\n\n### `+` 和 `*`")
        self.assertIn('href="#repr">一、<code>repr()</code></a>', output)
        self.assertIn('<code>+</code> 和 <code>*</code>', md.toc)
        sidebar = get_toc(md.toc_tokens)
        self.assertEqual(sidebar.items[0].title, "一、<code>repr()</code>")
        self.assertEqual(
            sidebar.items[0].children[0].title, "<code>+</code> 和 <code>*</code>"
        )

    def test_heading_ids_duplicate_suffixes_and_permalinks_are_unchanged(self):
        source = "[TOC]\n\n## `if` 判断\n\n## `if` 判断\n\n### Plain heading"
        fixed, output = render(source, permalink="🧀")
        original, _ = render(source, fixed=False, permalink="🧀")

        def ids(tokens):
            return [(t["id"], ids(t["children"])) for t in tokens]

        self.assertEqual(ids(fixed.toc_tokens), ids(original.toc_tokens))
        self.assertIn('class="headerlink"', output)
        self.assertNotIn('class="headerlink"', fixed.toc)
        self.assertNotIn("🧀", fixed.toc)

    def test_code_text_is_escaped_and_heading_links_are_not_nested(self):
        md, _ = render('[TOC]\n\n## [`<T> & x`](https://example.com) and **bold**')
        self.assertIn("<code>&lt;T&gt; &amp; x</code> and bold", md.toc)
        self.assertNotIn("https://example.com", md.toc)
        self.assertNotIn("<strong>", md.toc)

    def test_highlighted_code_keeps_text_without_attributes(self):
        md, _ = render('[TOC]\n\n## `#!python print("hello")`')
        self.assertIn('<code>print("hello")</code>', md.toc)
        self.assertNotIn('<span', md.toc)

    def test_explicit_toc_label_and_custom_id_take_precedence(self):
        md, _ = render('[TOC]\n\n## `repr()` {#custom data-toc-label="Short label"}')
        self.assertIn('href="#custom">Short label</a>', md.toc)
        self.assertEqual(md.toc_tokens[0]["name"], "Short label")

    def test_sidebar_without_marker_and_plain_headings(self):
        md, _ = render("## `return` statement")
        self.assertEqual(get_toc(md.toc_tokens).items[0].title, "<code>return</code> statement")
        plain = "[TOC]\n\n## Hello & goodbye\n\n### **Emphasis**"
        fixed, fixed_output = render(plain)
        original, original_output = render(plain, fixed=False)
        self.assertEqual(fixed_output, original_output)
        self.assertEqual(fixed.toc, original.toc)

    def test_hook_retains_toc_settings_and_is_idempotent(self):
        config = SimpleNamespace(
            markdown_extensions=["tables", "toc"],
            mdx_configs={"toc": {"permalink": "🧀", "toc_depth": 3}},
        )
        HOOK.on_config(config)
        extension = config.markdown_extensions[1]
        self.assertIsInstance(extension, HOOK.InlineCodeTocExtension)
        self.assertEqual(extension.getConfig("toc_depth"), 3)
        self.assertEqual(extension.getConfig("permalink"), "🧀")
        HOOK.on_config(config)
        self.assertIs(config.markdown_extensions[1], extension)


if __name__ == "__main__":
    unittest.main()

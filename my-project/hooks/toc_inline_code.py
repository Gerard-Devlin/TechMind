"""Keep inline code in both [TOC] and Material's per-page navigation."""

from html import escape
from html.parser import HTMLParser

from markdown.extensions.toc import TocExtension, TocTreeprocessor


class _CodeLabel(HTMLParser):
    """Retain code spans, but never copy links, attributes, or other heading HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.has_code = False

    def handle_starttag(self, tag, attrs):
        if tag == "code":
            self.has_code = True
            self.parts.append("<code>")

    def handle_endtag(self, tag):
        if tag == "code":
            self.parts.append("</code>")

    def handle_data(self, data):
        self.parts.append(escape(data, quote=False))


class InlineCodeTocTreeprocessor(TocTreeprocessor):
    def build_toc_div(self, toc_list):
        labels = {}

        def retain_code(items):
            for item in items:
                label = _CodeLabel()
                # An explicit TOC label takes precedence over the heading.
                label.feed(item.get("data-toc-label") or item.get("html", ""))
                label.close()
                if label.has_code:
                    # IDs have already been generated from the original plain text.
                    # MkDocs uses `name` as HTML for its sidebar AnchorLink title.
                    item["name"] = "".join(label.parts).strip()
                    labels["#" + item["id"]] = item["name"]
                retain_code(item["children"])

        retain_code(toc_list)
        div = super().build_toc_div(toc_list)
        for link in div.iter("a"):
            if link.get("href") in labels:
                # Let Markdown restore the safe label after serializing [TOC].
                link.text = self.md.htmlStash.store(labels[link.get("href")])
        return div


class InlineCodeTocExtension(TocExtension):
    TreeProcessorClass = InlineCodeTocTreeprocessor


def on_config(config):
    """Replace only the configured TOC extension, retaining all of its options."""
    for index, extension in enumerate(config.markdown_extensions):
        if extension == "toc":
            config.markdown_extensions[index] = InlineCodeTocExtension(
                **config.mdx_configs.get("toc", {})
            )
    return config

"""Multi-format document loader: markdown, text, HTML, PDF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DocumentId = str


@dataclass
class Document:
    """Normalized document with metadata."""

    id: DocumentId
    content: str
    source: str
    section_heading: str = ""
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "section_heading": self.section_heading,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


class DocumentLoader:
    """Load and normalize documents from disk. Supports: .txt, .md, .html, .pdf."""

    EXT_LOADERS = {
        ".txt": "text",
        ".md": "markdown",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
    }

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        ext = path.suffix.lower()
        loader_type = self.EXT_LOADERS.get(ext)
        if loader_type is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return getattr(self, f"_load_{loader_type}")(path)

    def load_dir(self, root: str | Path, glob_pattern: str = "*") -> list[Document]:
        root = Path(root)
        docs = []
        for path in root.rglob(glob_pattern):
            if path.is_file() and path.suffix.lower() in self.EXT_LOADERS:
                try:
                    docs.append(self.load(path))
                except Exception as exc:
                    print(f"[WARN] Failed to load {path}: {exc}")
        return docs

    def _load_text(self, path: Path) -> Document:
        return Document(
            id=self._make_id(path),
            content=path.read_text(encoding="utf-8").strip(),
            source=str(path),
        )

    def _load_markdown(self, path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        m = re.match(r"^#+\s+(.+)", content.strip())
        heading = m.group(1).strip() if m else ""
        return Document(
            id=self._make_id(path),
            content=content.strip(),
            source=str(path),
            section_heading=heading,
        )

    def _load_html(self, path: Path) -> Document:
        html = path.read_text(encoding="utf-8")
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return Document(id=self._make_id(path), content=text, source=str(path))

    def _load_pdf(self, path: Path) -> Document:
        """Extract text from PDF. Fails gracefully for image-only PDFs."""
        try:
            import pymupdf
        except ImportError:
            raise ImportError("pymupdf required for PDF loading: pip install pymupdf")

        doc = pymupdf.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(Document(
                id=f"{self._make_id(path)}_p{i+1}",
                content=text.strip() if text.strip() else "[IMAGE-ONLY]",
                source=str(path),
                page_number=i + 1,
                metadata={"total_pages": len(doc), "image_only": not bool(text.strip())},
            ))
        doc.close()
        combined = "\n\n".join(p.content for p in pages)
        first_meta = pages[0].metadata if pages else {}
        return Document(
            id=self._make_id(path),
            content=combined,
            source=str(path),
            metadata=first_meta,
        )

    @staticmethod
    def _make_id(path: Path) -> str:
        import hashlib
        return hashlib.sha256(str(path).encode()).hexdigest()[:16]

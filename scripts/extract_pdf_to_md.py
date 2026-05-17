import sys
import fitz
from pathlib import Path


def extract_to_markdown(pdf_path: str, out_path: str) -> None:
    doc = fitz.open(pdf_path)
    lines = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Ensure consistent newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines.append(f"\n\n<!-- Page {i+1} -->\n\n")
        lines.append(text)
    content = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: extract_pdf_to_md.py <input.pdf> <output.md>")
        sys.exit(1)
    extract_to_markdown(sys.argv[1], sys.argv[2])

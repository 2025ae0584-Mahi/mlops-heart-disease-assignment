import pathlib
import re

import markdown

SRC = pathlib.Path("docs/final_report.md")
DST = pathlib.Path("REPORT.html")

text = SRC.read_text(encoding="utf-8")

body = markdown.markdown(
    text,
    extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
)

# Rewrite relative image srcs so they resolve from DST's location, not SRC's.
img_dir = SRC.parent.resolve()
dst_dir = DST.parent.resolve()


def _rewrite_src(match: re.Match) -> str:
    src = match.group(1)
    if re.match(r"^(https?:)?//|^data:|^/", src):
        return match.group(0)
    relative = pathlib.PurePosixPath(
        pathlib.Path(img_dir / src).resolve().relative_to(dst_dir).as_posix()
    )
    return match.group(0).replace(src, str(relative))


body = re.sub(r'<img[^>]*\bsrc="([^"]+)"', _rewrite_src, body)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MLOps Assignment 1: Final Technical Report</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #1a1a1a; }}
pre, code {{ background: #f5f5f5; border-radius: 4px; }}
pre {{ padding: 1rem; overflow-x: auto; }}
code {{ padding: 0.1rem 0.3rem; }}
pre code {{ padding: 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 0.5rem; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; }}
.mermaid {{ background: #fff; }}
</style>
</head>
<body>
{body}
<script>
  document.querySelectorAll('pre code.language-mermaid, code.language-mermaid').forEach((el) => {{
    const pre = el.closest('pre');
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = el.textContent;
    pre.replaceWith(div);
  }});
  mermaid.initialize({{ startOnLoad: true }});
</script>
</body>
</html>
"""

DST.write_text(html, encoding="utf-8")
print(f"Wrote {DST.resolve()}")

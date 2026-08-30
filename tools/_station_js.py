"""Replace the page's script block with the data-driven version.

One-shot: run once to convert the hand-written page into one that reads its
scenario from a JSON block. After that, `build_station.py` + `_patch_station.py`
keep it in sync with the engine.
"""

import pathlib
import sys

PAGE = pathlib.Path(sys.argv[1])
JS = pathlib.Path(__file__).parent / "station.js"

html = PAGE.read_text(encoding="utf-8")
start = html.index("<script>\n(function () {")

block = (
    '<script id="scenario" type="application/json">{}</script>\n'
    "<script>\n" + JS.read_text(encoding="utf-8") + "\n</script>\n"
)
PAGE.write_text(html[:start] + block, encoding="utf-8")
print(f"script block replaced in {PAGE.name}")

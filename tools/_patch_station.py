"""Splice the generated scenario into the station page.

    python tools/build_station.py > tools/station_data.json
    python tools/_patch_station.py

Kept as a file rather than an inline command because the page's JavaScript is
full of apostrophes and shell heredocs mangle them.
"""

import json
import pathlib
import sys

PAGE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
DATA = pathlib.Path(__file__).parent / "station_data.json"

if PAGE is None or not PAGE.exists():
    raise SystemExit("usage: python tools/_patch_station.py <path-to-triage-station.html>")

html = PAGE.read_text(encoding="utf-8")
scenario = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), separators=(",", ":"))

marker = '<script id="scenario" type="application/json">'
if marker not in html:
    raise SystemExit("scenario block not found in the page")

start = html.index(marker) + len(marker)
end = html.index("</script>", start)
html = html[:start] + scenario + html[end:]

PAGE.write_text(html, encoding="utf-8")
print(f"injected {len(scenario)} chars of engine-generated scenario into {PAGE.name}")

"""Render the Chapter 2 diagrams from their Excalidraw element lists.

Each ``*.json`` file in this directory is the element list that was drawn with the
Excalidraw MCP view (shapes with inline ``label``s, arrows as point lists, camera
pseudo-elements). This script turns every list into

* ``docs/figures/<name>.svg``: a clean vector figure embedded by the thesis Markdown, and
* ``docs/figures/<name>.excalidraw``: a native scene that opens in excalidraw.com for editing.

Usage: ``python docs/figures/src/render_diagrams.py`` (standard library only).
"""

from __future__ import annotations

import json
import math
import random
from html import escape
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
FONT = "'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif"
PAD = 30
LINE = 1.25  # line height, as in Excalidraw


def _drawable(elements: list[dict]) -> list[dict]:
    return [e for e in elements if e["type"] not in ("cameraUpdate", "delete", "restoreCheckpoint")]


def _text_width(text: str, size: float) -> float:
    # Combining marks (hats, dots, tildes) take no horizontal space.
    visible = [c for c in text if not 0x0300 <= ord(c) <= 0x036F]
    return len(visible) * size * 0.56


def _arrow_points(e: dict) -> list[tuple[float, float]]:
    return [(e["x"] + dx, e["y"] + dy) for dx, dy in e["points"]]


def _midpoint(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Point halfway along a polyline (where Excalidraw puts arrow labels)."""
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    half = sum(seg) / 2
    for (a, b), s in zip(zip(pts, pts[1:]), seg):
        if half <= s and s > 0:
            t = half / s
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        half -= s
    return pts[-1]


def _bbox(elements: list[dict]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for e in elements:
        if e["type"] == "arrow":
            for x, y in _arrow_points(e):
                xs.append(x)
                ys.append(y)
            if "label" in e:
                cx, cy = _midpoint(_arrow_points(e))
                size = e["label"].get("fontSize", 20)
                half = max(_text_width(s, size) for s in e["label"]["text"].split("\n")) / 2
                xs += [cx - half, cx + half]
        elif e["type"] == "text":
            lines = e["text"].split("\n")
            xs += [e["x"], e["x"] + max(_text_width(s, e["fontSize"]) for s in lines)]
            ys += [e["y"], e["y"] + len(lines) * e["fontSize"] * LINE]
        else:
            xs += [e["x"], e["x"] + e["width"]]
            ys += [e["y"], e["y"] + e["height"]]
    return min(xs), min(ys), max(xs), max(ys)


def _tspans(text: str, cx: float, cy: float, size: float, anchor: str = "middle") -> str:
    lines = text.split("\n")
    y0 = cy - (len(lines) - 1) * size * LINE / 2
    spans = "".join(
        f'<tspan x="{cx:.1f}" y="{y0 + k * size * LINE:.1f}">{escape(s)}</tspan>'
        for k, s in enumerate(lines)
    )
    return (
        f'<text font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
        f'dominant-baseline="central" fill="#1e1e1e">{spans}</text>'
    )


def to_svg(elements: list[dict]) -> tuple[str, list[str]]:
    """SVG markup plus a list of labels that may overflow their shape."""
    els = _drawable(elements)
    x0, y0, x1, y1 = _bbox(els)
    x0, y0, x1, y1 = x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD
    w, h = x1 - x0, y1 - y0
    warnings: list[str] = []
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}" '
        f'width="{w:.0f}" height="{h:.0f}" role="img">',
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
        'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        'fill="context-stroke"/></marker></defs>',
        f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{h:.0f}" fill="#ffffff"/>',
    ]
    for e in els:
        stroke = e.get("strokeColor", "#1e1e1e")
        fill = e.get("backgroundColor", "transparent")
        fill = "none" if fill == "transparent" else fill
        sw = e.get("strokeWidth", 2)
        dash = ' stroke-dasharray="8 6"' if e.get("strokeStyle") == "dashed" else ""
        op = f' opacity="{e["opacity"] / 100:.2f}"' if "opacity" in e else ""
        common = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}{op}'
        t = e["type"]
        if t == "rectangle":
            rx = 12 if e.get("roundness") else 0
            out.append(
                f'<rect x="{e["x"]}" y="{e["y"]}" width="{e["width"]}" height="{e["height"]}" '
                f'rx="{rx}" {common}/>'
            )
        elif t == "ellipse":
            out.append(
                f'<ellipse cx="{e["x"] + e["width"] / 2}" cy="{e["y"] + e["height"] / 2}" '
                f'rx="{e["width"] / 2}" ry="{e["height"] / 2}" {common}/>'
            )
        elif t == "diamond":
            cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            pts = f'{cx},{e["y"]} {e["x"] + e["width"]},{cy} {cx},{e["y"] + e["height"]} {e["x"]},{cy}'
            out.append(f'<polygon points="{pts}" {common}/>')
        elif t == "arrow":
            pts = _arrow_points(e)
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            head = ' marker-end="url(#ah)"' if e.get("endArrowhead", "arrow") else ""
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{sw}"'
                f'{dash} stroke-linejoin="round"{head}/>'
            )
        elif t == "text":
            size = e["fontSize"]
            lines = e["text"].split("\n")
            spans = "".join(
                f'<tspan x="{e["x"]}" y="{e["y"] + k * size * LINE}">{escape(s)}</tspan>'
                for k, s in enumerate(lines)
            )
            out.append(
                f'<text font-family="{FONT}" font-size="{size}" dominant-baseline="hanging" '
                f'fill="{stroke}">{spans}</text>'
            )
        label = e.get("label")
        if not label:
            continue
        size = label.get("fontSize", 20)
        text = label["text"]
        lw = max(_text_width(s, size) for s in text.split("\n"))
        lh = len(text.split("\n")) * size * LINE
        if t == "arrow":
            cx, cy = _midpoint(_arrow_points(e))
            out.append(
                f'<rect x="{cx - lw / 2 - 4:.1f}" y="{cy - lh / 2 - 2:.1f}" width="{lw + 8:.1f}" '
                f'height="{lh + 4:.1f}" fill="#ffffff" opacity="0.9"/>'
            )
        else:
            cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            room = e["width"] * (0.7 if t == "diamond" else 1.0) - 12
            if lw > room or lh > e["height"] - 6:
                warnings.append(f"{e['id']}: label {lw:.0f}x{lh:.0f} in {e['width']}x{e['height']}")
        out.append(_tspans(text, cx, cy, size))
    out.append("</svg>")
    return "\n".join(out), warnings


def to_excalidraw(elements: list[dict]) -> dict:
    """Native Excalidraw scene: inline labels become bound text elements."""
    rng = random.Random(7)
    scene: list[dict] = []

    def base(e: dict) -> dict:
        return {
            "id": e["id"], "type": e["type"], "x": e["x"], "y": e["y"],
            "width": e.get("width", 0), "height": e.get("height", 0), "angle": 0,
            "strokeColor": e.get("strokeColor", "#1e1e1e"),
            "backgroundColor": e.get("backgroundColor", "transparent"),
            "fillStyle": e.get("fillStyle", "solid"), "strokeWidth": e.get("strokeWidth", 2),
            "strokeStyle": e.get("strokeStyle", "solid"), "roughness": 1,
            "opacity": e.get("opacity", 100), "groupIds": [], "frameId": None,
            "roundness": e.get("roundness"), "seed": rng.randrange(1, 2**31),
            "version": 1, "versionNonce": rng.randrange(1, 2**31), "isDeleted": False,
            "boundElements": [], "updated": 1, "link": None, "locked": False,
        }

    def text_el(tid: str, text: str, size: float, cx: float, cy: float, container: str | None) -> dict:
        lines = text.split("\n")
        w = max(_text_width(s, size) for s in lines)
        h = len(lines) * size * LINE
        el = base({"id": tid, "type": "text", "x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h})
        el.update({
            "text": text, "originalText": text, "fontSize": size, "fontFamily": 1,
            "textAlign": "center", "verticalAlign": "middle", "containerId": container,
            "lineHeight": LINE, "autoResize": True, "roundness": None,
        })
        return el

    for e in _drawable(elements):
        el = base(e)
        if e["type"] == "text":
            el.update({
                "text": e["text"], "originalText": e["text"], "fontSize": e["fontSize"],
                "fontFamily": 1, "textAlign": "left", "verticalAlign": "top",
                "containerId": None, "lineHeight": LINE, "autoResize": True,
            })
            lines = e["text"].split("\n")
            el["width"] = max(_text_width(s, e["fontSize"]) for s in lines)
            el["height"] = len(lines) * e["fontSize"] * LINE
        if e["type"] == "arrow":
            el.update({
                "points": e["points"], "lastCommittedPoint": None, "startBinding": None,
                "endBinding": None, "startArrowhead": None,
                "endArrowhead": e.get("endArrowhead", "arrow"), "elbowed": False,
            })
        scene.append(el)
        label = e.get("label")
        if label:
            tid = f"{e['id']}_label"
            if e["type"] == "arrow":
                cx, cy = _midpoint(_arrow_points(e))
            else:
                cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            scene.append(text_el(tid, label["text"], label.get("fontSize", 20), cx, cy, e["id"]))
            el["boundElements"].append({"type": "text", "id": tid})
    return {
        "type": "excalidraw", "version": 2, "source": "docs/figures/src/render_diagrams.py",
        "elements": scene, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
        "files": {},
    }


def main() -> None:
    for src in sorted(SRC.glob("*.json")):
        elements = json.loads(src.read_text(encoding="utf-8"))
        svg, warnings = to_svg(elements)
        (OUT / f"{src.stem}.svg").write_text(svg, encoding="utf-8")
        (OUT / f"{src.stem}.excalidraw").write_text(
            json.dumps(to_excalidraw(elements), indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{src.stem}: svg + excalidraw" + "".join(f"\n  overflow? {w}" for w in warnings))


if __name__ == "__main__":
    main()

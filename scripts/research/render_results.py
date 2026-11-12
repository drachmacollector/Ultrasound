"""
scripts/research/render_results.py  (R6 A: RESULTS.md from results/json only)

Fills docs/research/templates/RESULTS.template.md and writes docs/research/RESULTS.md.

Placeholders:
  {{E2.prior.seen_machines.auroc}}           value at that key path of results/json/E2.json
  {{E1.other_share_gap|pct}}                 modifiers after '|': pct (x100, 1 dp, '%' / ' pp' for deltas),
  {{R4_localisation.methods|join}}           dp=N (decimals, default 3), bare (no n / p suffix), join (list)
  {{@deviations}}                            every RESEARCH_LOG line starting 'deviation:', grouped by entry

List indices are written as path parts (``foo.0.bar``). A key that contains '|' is written with
'~' instead (``R4_localisation.p3a.gradcam.convnext_tiny.fpdb_test~Head``). Values render as:
  Estimate (point, lo, hi, n_images, n_clusters)   0.735 [0.714, 0.751] (n = 4,153 images, 712 patients)
  PairedEstimate (delta, lo, hi, p_two_sided)      +0.137 [+0.122, +0.154], p = 0.0005
  number / string / bool                           as is (numbers to dp)

A missing key or JSON file is a hard error; ``--allow-missing`` renders it as ⟨pending⟩ instead.
The output header lists the git hash, the render time and the SHA-256 of every JSON consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

JSON_DIR = Path("results/json")
TEMPLATE = Path("docs/research/templates/RESULTS.template.md")
OUT = Path("docs/research/RESULTS.md")
LOG = Path("docs/research/RESEARCH_LOG.md")
PENDING = "⟨pending⟩"
PLACEHOLDER = re.compile(r"\{\{\s*([^}|]+?)\s*(?:\|\s*([^}]*?)\s*)?\}\}")


class MissingKey(KeyError):
    pass


def _num(x: float, dp: int, signed: bool = False, pct: bool = False) -> str:
    v = 100 * x if pct else x
    d = 1 if pct else dp
    return f"{v:+,.{d}f}" if signed else f"{v:,.{d}f}"


def fmt_value(v, mods: dict[str, str]) -> str:
    dp = int(mods.get("dp", 3))
    pct = "pct" in mods
    bare = "bare" in mods
    if isinstance(v, dict) and {"delta", "lo", "hi"} <= set(v):
        unit = " pp" if pct else ""
        s = (f"{_num(v['delta'], dp, True, pct)}{unit} [{_num(v['lo'], dp, True, pct)}, "
             f"{_num(v['hi'], dp, True, pct)}]")
        if not bare and v.get("p_two_sided") is not None:
            s += f", p = {v['p_two_sided']:.2g}"
        return s
    if isinstance(v, dict) and {"point", "lo", "hi"} <= set(v):
        if v["point"] is None:
            return "n/a"
        unit = "%" if pct else ""
        s = f"{_num(v['point'], dp, pct=pct)}{unit} [{_num(v['lo'], dp, pct=pct)}, {_num(v['hi'], dp, pct=pct)}]"
        if not bare and v.get("n_images") is not None:
            s += f" (n = {v['n_images']:,} images, {v['n_clusters']:,} patients)"
        return s
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return _num(v, dp, pct=pct) + ("%" if pct else "")
    if isinstance(v, list) and "join" in mods:
        return ", ".join(str(x) for x in v)
    if isinstance(v, str):
        return v
    raise TypeError(f"cannot render {type(v).__name__}: {str(v)[:80]}")


def parse_mods(s: str | None) -> dict[str, str]:
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            k, _, val = part.partition("=")
            out[k.strip()] = val.strip()
    return out


class Store:
    def __init__(self, root: Path = JSON_DIR):
        self.root, self.cache, self.used = root, {}, {}

    def get(self, path: str):
        parts = path.split(".")
        name = parts[0]
        if name not in self.cache:
            f = self.root / f"{name}.json"
            if not f.exists():
                raise MissingKey(f"{path}: no {f}")
            raw = f.read_bytes()
            self.cache[name] = json.loads(raw.decode("utf-8"))
            self.used[f.as_posix()] = hashlib.sha256(raw).hexdigest()
        v = self.cache[name]
        for p in parts[1:]:
            p = p.replace("~", "|")
            if isinstance(v, list) and p.lstrip("-").isdigit():
                idx = int(p)
                if not -len(v) <= idx < len(v):
                    raise MissingKey(f"{path}: index {p} out of range")
                v = v[idx]
            elif isinstance(v, dict) and p in v:
                v = v[p]
            else:
                raise MissingKey(f"{path}: no key {p!r}")
        return v


def deviations(log: Path = LOG) -> str:
    """'deviation:' lines from the research log, grouped under their entry heading."""
    out, head = [], None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
        elif line.startswith("deviation:"):
            if head is not None:
                out.append(f"\n**{head}**\n")
                head = None
            out.append(f"- {line[len('deviation:'):].strip()}")
    return "\n".join(out).strip() or "None recorded."


def render(template: str, store: Store, allow_missing: bool = False, log: Path = LOG) -> tuple[str, list[str]]:
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key, mods = m.group(1).strip(), parse_mods(m.group(2))
        if key == "@deviations":
            return deviations(log)
        try:
            return fmt_value(store.get(key), mods)
        except MissingKey as e:
            if not allow_missing:
                raise
            missing.append(str(e))
            return PENDING

    return PLACEHOLDER.sub(sub, template), missing


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    store = Store()
    body, missing = render(Path(a.template).read_text(encoding="utf-8"), store, a.allow_missing)
    header = ["<!-- AUTO-GENERATED by scripts/research/render_results.py; do not edit by hand -->",
              f"<!-- git {git_hash()} | rendered {time.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"{len(missing)} pending -->"]
    header += [f"<!-- {p} sha256 {h} -->" for p, h in sorted(store.used.items())]
    Path(a.out).write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")
    print(f"wrote {a.out} ({len(store.used)} JSON files, {len(missing)} pending)")


if __name__ == "__main__":
    main()

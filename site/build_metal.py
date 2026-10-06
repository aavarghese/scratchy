#!/usr/bin/env python3
"""Generate site/_site/metal.html from site/data/metal/*.json.

Each data file is one run of scripts/bench_metal_matrix.sh, copied in
unedited and named <machine>-<YYYY-MM-DDTHHMMSSZ>-<sha8>.json (older runs:
<machine>-<YYYY-MM-DD>-<sha8>.json). The page reads them at
site-build time, so it cannot drift from what the runner measured, and a file
missing what the page needs fails the build rather than rendering a blank.

Per machine, each model shows its newest run; older runs of the same model
stay visible in that model's history table, so a re-run after a Metal change
reads as a before/after.

Standalone: site/build_metal.py [out.html] [data_dir]
"""
import html
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data" / "metal"
REPO = "https://github.com/AI-native-Systems-Research/scratchy"

# Every engine the runner knows, in page order, as (key, label). Its JSON fields
# are cache_ladder / scaling for scratchy and cache_ladder_<key> / scaling_<key>
# for the rest; older runs used the same pattern for mlx_lm and ollama alone.
ENGINE_ORDER = [
    ("scratchy", "scratchy"),
    ("omlx_tq", "oMLX TurboQuant"),
    ("omlx", "oMLX"),
    ("mlx_lm", "mlx-lm"),
    ("vllm_metal", "vllm-metal"),
    ("ollama", "ollama"),
    ("llama_cpp", "llama.cpp"),
    ("mistralrs", "mistral.rs"),
]
LABEL = dict(ENGINE_ORDER)
# The engine a headline comparison uses: the first present. oMLX with TurboQuant
# meets scratchy's TurboQuant default like for like; then oMLX as installed; then
# mlx-lm, which runs the same MLX checkpoint; then the rest.
RIVAL_PREFERENCE = ["omlx_tq", "omlx", "mlx_lm", "vllm_metal", "ollama", "llama_cpp", "mistralrs"]


def fields(key):
    """An engine's (startup ladder, scaling) JSON fields."""
    return ("cache_ladder", "scaling") if key == "scratchy" else (f"cache_ladder_{key}", f"scaling_{key}")


def engines_in(m):
    """The engines this model entry has numbers for, in page order: (key, label,
    ladder, scaling cells). scratchy is always first, numbers or not."""
    out = []
    for key, label in ENGINE_ORDER:
        lk, sk = fields(key)
        if key == "scratchy" or m.get(lk) or m.get(sk):
            out.append((key, label, m.get(lk), m.get(sk) or []))
    return out


def served_as(m, key):
    """What an engine served for this model, for its row label: the ollama tag
    or GGUF file it loaded, with its quantization when it reported one."""
    e = (m.get("engine_models") or {}).get(key) or {}
    if not e and key == "ollama" and m.get("ollama"):        # runs before engine_models
        e = {"model": m["ollama"].get("tag"), "quantization": m["ollama"].get("quantization")}
    model = (e.get("model") or "").split(":")[-1] if key in ("llama_cpp", "mistralrs") else e.get("model")
    return ", ".join(x for x in (model if key in ("ollama", "llama_cpp", "mistralrs") else None,
                                 e.get("quantization")) if x)

REQUIRED = {
    "": ("schema", "generated_utc", "generator", "machine", "repo", "config", "models"),
    "machine": ("chip", "cores_total", "memory_gb", "macos"),
    "repo": ("sha",),
    "config": ("scenarios",),
}
MODEL_REQUIRED = ("stem", "model_id", "built")


def esc(s):
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------- load

def load(data_dir):
    """Every run file, validated. Exits naming each problem, so one build
    reports all of them rather than the first."""
    runs, errors = [], []
    for f in sorted(Path(data_dir).glob("*.json")):
        try:
            run = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{f.name}: not valid JSON ({e})")
            continue
        if not isinstance(run, dict):
            errors.append(f"{f.name}: not a run object")
            continue
        before = len(errors)
        for block, keys in REQUIRED.items():
            obj = run if not block else run.get(block)
            if not isinstance(obj, dict):
                errors.append(f"{f.name}: missing {block}")
                continue
            for k in keys:
                if obj.get(k) in (None, ""):
                    errors.append(f"{f.name}: missing {block + '.' if block else ''}{k}")
        if run.get("schema") not in (None, 2):
            errors.append(f"{f.name}: schema {run.get('schema')!r}, this page reads schema 2")
        models = run.get("models")
        if not isinstance(models, list):
            errors.append(f"{f.name}: models is not a list")
        else:
            for i, m in enumerate(models):
                missing = [k for k in MODEL_REQUIRED if not isinstance(m, dict) or k not in m]
                if missing:
                    errors.append(f"{f.name}: models[{i}] missing {', '.join(missing)}")
        if len(errors) > before:          # the name check needs the fields above
            continue
        # The start time keeps two runs of one commit on one day apart; names
        # from before it carry only the date.
        machine, utc, sha8 = slug(run['machine']['chip']), run['generated_utc'], str(run['repo']['sha'])[:8]
        want = f"{machine}-{utc[:10]}T{utc[11:19].replace(':', '')}Z-{sha8}.json"
        if f.name not in (want, f"{machine}-{utc[:10]}-{sha8}.json"):
            errors.append(f"{f.name}: name it {want} (<machine>-<start time>-<sha8>.json)")
        run["_file"] = f
        runs.append(run)
    if errors:
        for e in errors:
            print(f"metal page: {e}", file=sys.stderr)
        sys.exit(1)
    return runs


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


# ------------------------------------------------------------------ numbers

def med(reps, scenario, field):
    vals = [r[field] for r in reps or [] if r.get("scenario") == scenario and r.get(field) is not None]
    return statistics.median(vals) if vals else None


def cell(cells, axis, rung):
    for c in cells or []:
        if c.get("axis") == axis and c.get("rung") == rung:
            return c
    return None


def lfmt(ladder, scenario, field, nd=0, scale=1):
    """A startup median, or why there is none: launches that reached ready but
    never streamed a token say so instead of showing a dash."""
    v = med(ladder, scenario, field)
    if v is None and field != "t_ready_s" and med(ladder, scenario, "t_ready_s") is not None:
        return "no first token"
    return fmt(v, nd, scale)


def served(m):
    """Whether scratchy produced any number for this model."""
    return bool(m.get("built") and (m.get("cache_ladder") or m.get("warm_serving") or m.get("scaling")))


def fmt(v, nd=0, scale=1):
    return "—" if v is None else f"{v * scale:,.{nd}f}"


TIMINGS = ("median_ttft_ms", "median_tpot_ms")
NO_STREAM = "no stream"


def streamed(c):
    """Whether a serving cell's answer actually streamed. A zero median TPOT
    means the tokens arrived in one piece (or there was only one), so neither
    timing was measured: its TTFT is the whole answer's time, not the first
    token's. ollama's gemma4 does this."""
    return bool(c) and (c.get("median_tpot_ms") or 0) > 0


def timing(c, metric):
    """A cell's value, or None where it was not really measured."""
    if not c or (metric in TIMINGS and not streamed(c)):
        return None
    return c.get(metric)


def untimed(c):
    """Requests in a cell that arrived in one piece and so were left out of its
    TTFT/TPOT (bench serve's `unstreamed_requests`; absent, so 0, in runs
    made before it existed)."""
    return (c or {}).get("unstreamed_requests") or 0


PARTIAL = "†"


def partial(c, metric):
    """Whether a timing is real but taken from only some of the cell's requests."""
    return metric in TIMINGS and streamed(c) and untimed(c) > 0


def untimed_note(c):
    return f"{untimed(c)} of {c.get('completed', '?')} untimed"


def cfmt(c, metric, nd=0):
    """A cell's value as text: says "no stream" rather than show a timing the
    engine never produced, and marks one taken from only some requests."""
    if c and metric in TIMINGS and not streamed(c):
        return NO_STREAM
    return fmt(timing(c, metric), nd) + (PARTIAL if partial(c, metric) else "")


def nice_max(v):
    """Smallest 1/2/5 x 10^k at or above v, so ticks land on round numbers."""
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


# ------------------------------------------------------------------- render

def tip(rows):
    """A tooltip payload: [[value, label, series-class], ...]. Rendered by the
    page's script with textContent, never innerHTML."""
    return esc(json.dumps(rows, separators=(",", ":")))


def info(body, label="", align="bottom"):
    """A Carbon toggletip: an (i) button opening a short note, so explanations
    sit where the question comes up instead of pushing the numbers down.
    Carbon fixes its width at 18rem, so keep `body` to a few sentences. The
    body is a <span>, not a <p>: toggletips sit inside paragraphs, and a <p>
    in a <p> makes the parser close the outer one and spill the text out.
    No `autoalign`: Carbon then calls scrollIntoView on the button every time
    it renders, page load included, so a page of toggletips jumps to the last
    one (Firefox lands at the bottom). A fixed alignment never scrolls."""
    return (f'<cds-toggletip class="minfo" alignment="{align}" button-label="{esc(label or "What this means")}">'
            f'{esc(label)}<span slot="body-text" class="minfo-body">{body}</span></cds-toggletip>')


def fold(title, body):
    """A collapsed Carbon accordion item, for detail most readers skip."""
    return (f'<cds-accordion class="mfold"><cds-accordion-item title="{esc(title)}">'
            f'{body}</cds-accordion-item></cds-accordion>')


# Run-to-run spread on one machine at one commit: Nick's two M1 Max runs of
# af009bf1 agreed within about 2%, so a smaller change is noise, not a result.
NOISE = 0.03


def delta(new, old, higher_better):
    """A change badge against the previous run: arrow plus percent, the arrow
    coloured by better/worse so colour is never the only signal; "≈" inside
    the noise band; nothing when either side is missing."""
    if new is None or old is None or old == 0:
        return ""
    ch = (new - old) / old
    if abs(ch) < NOISE:
        return '<span class="delta" title="within 3% of the previous run">≈</span>'
    better = (ch > 0) == higher_better
    word = "better" if better else "worse"
    return (f'<span class="delta" title="{word} than the previous run">'
            f'<i class="{word}">{"▲" if ch > 0 else "▼"}</i>{abs(ch) * 100:.0f}%</span>')


def summary_raw(m, lk, sk):
    """The summary row's numbers, unformatted, for comparing two runs."""
    ladder, one = m.get(lk), cell(m.get(sk), "conc", 1)
    return [med(ladder, "frozen", "t_ready_s"), med(ladder, "cold", "t_ready_s"),
            med(ladder, "cold", "ttft_from_send_s"), med(ladder, "warm", "ttft_from_send_s"),
            timing(one, "median_ttft_ms"), timing(one, "median_tpot_ms"),
            timing(one, "output_throughput"), med(ladder, "cold", "peak_rss_mib")]


# Only tok/s is better higher; every time and memory column is better lower.
SUMMARY_HIGHER_BETTER = [False, False, False, False, False, False, True, False]


STARTUP_INFO = info(
    "<b>frozen</b>: first launch after memory and scratchy's weights cache are cleared. "
    "<b>cold</b>: a normal relaunch. Both run until the server is ready, no request "
    "included. <b>1st req</b>: the cold server's first request, send to first token. "
    "<b>warm</b>: the median over many requests once it is running. ollama reports ready "
    "before loading the model, so its load time lands in 1st req.")
TIMING_INFO = info(
    "<b>no stream</b>: the answer arrived in one piece, so TTFT and TPOT could not be "
    "timed. <b>†</b>: some answers did; the timing comes from the rest (hover for how "
    "many). mlx-lm and ollama can stop answers early, which flatters tok/s.")
RSS_INFO = info("The server process at its peak. For ollama, <code>ollama serve</code> "
                "only; its runner process is not counted.")
USERS_INFO = info("Requests in flight at once. An engine may batch fewer than it is offered.")


# These sit in the section's subtitle, not in the table headings: the table
# scrolls sideways inside its own box, which would clip a popover opened from it.


def settings(run):
    """The run settings that change scratchy's numbers on their own: its serve
    flags and KV cache setting, as the runner recorded them."""
    c = run["config"]
    return {"serve flags": c.get("scratchy_serve_args") or "none",
            "KV cache": c.get("kv_cache_dtype") or "default"}


def base_of(run, m):
    """The base shape this model ran at: the run's, with the users-at-once the
    runner lowered for a model whose KV cache would not fit more."""
    base = dict((run["config"].get("scaling") or {}).get("base") or {})
    limits = m.get("scaling_limits") or {}
    if base and limits.get("base_conc") is not None:
        base["conc"] = limits["base_conc"]
    return base


def shape(run, m):
    base = base_of(run, m)
    return "/".join(str(base.get(k, "?")) for k in ("input", "output", "conc")) if base else "not recorded"


def conditions(run, m):
    """Everything else that can move a run-to-run number besides scratchy's
    code: the base shape, the other engines' releases, whether the run was
    pinned, macOS, and the model files. Unrecorded on either side never counts
    as a change, so runs from before a field existed do not all warn."""
    files = (m.get("model_files") or {}).get("revision")
    return {"base shape (in/out/users)": shape(run, m), "engine versions": engine_versions(run),
            "pinned": pinned(run).split(":")[0], "macOS": run["machine"].get("macos") or "not recorded",
            "model files": files[:8] if files else "not recorded"}


def settings_changed(run, prev):
    """Human-readable differences in settings between two runs, or ""."""
    now, then = settings(run), settings(prev)
    return "; ".join(f"{k} {then[k]} then, {now[k]} now" for k in now if now[k] != then[k])


def conditions_changed(run, m, prev):
    """Like settings_changed, for the conditions around scratchy, or ""."""
    now, then = conditions(run, m), conditions(prev[0], prev[1])
    return "; ".join(f"{k} {then[k]} then, {now[k]} now" for k in now
                     if now[k] != then[k] and "not recorded" not in (now[k], then[k]))


def summary_table(m, run, prev=None):
    base = base_of(run, m)
    shape = f"{base.get('input', '?')} in / {base.get('output', '?')} out" if base else ""
    rows = []
    for key, label, ladder, cells in engines_in(m):
        one = cell(cells, "conc", 1)
        cls = "s1" if key == "scratchy" else "s0"
        if key == "scratchy" and not served(m):
            why = ("build failed" if not m.get("built")
                   else "its answers did not match mlx-lm's, so no engine was timed"
                   if m.get("parity_mlx_lm") is False else "the server never served")
            rows.append(f'<tr><th scope="row"><span class="key {cls}"></span>{label}</th>'
                        f'<td colspan="8" class="gap">No numbers: {why}. See the run\'s raw logs.</td></tr>')
            continue
        name = esc(label)
        detail = served_as(m, key)
        if detail:
            name += f' <span class="dim">{esc(detail)}</span>'
        vals = [
            lfmt(ladder, "frozen", "t_ready_s", 2),
            lfmt(ladder, "cold", "t_ready_s", 2),
            lfmt(ladder, "cold", "ttft_from_send_s", 2),
            lfmt(ladder, "warm", "ttft_from_send_s", 0, 1000),
            cfmt(one, "median_ttft_ms"),
            cfmt(one, "median_tpot_ms", 1),
            fmt(timing(one, "output_throughput"), 1),
            fmt(med(ladder, "cold", "peak_rss_mib")),
        ]
        badges = [""] * len(vals)
        if key == "scratchy" and prev is not None:
            lk, sk = fields("scratchy")
            badges = [delta(n, o, hb) for n, o, hb in
                      zip(summary_raw(m, lk, sk), summary_raw(prev[1], lk, sk), SUMMARY_HIGHER_BETTER)]
        rows.append(f'<tr><th scope="row"><span class="key {cls}"></span>{name}</th>'
                    + "".join(f"<td>{v}{b}</td>" for v, b in zip(vals, badges)) + "</tr>")
    vs = ""
    if prev is not None:
        vs = (f' Under scratchy: change since its previous run here ({esc(prev[0]["generated_utc"][:10])}, '
              f'<code>{esc(str(prev[0]["repo"]["sha"])[:8])}</code>); ≈ is within 3%.')
        changed = settings_changed(run, prev[0])
        if changed:
            vs += (f' <b>Settings differ from that run</b> ({esc(changed)}), so a change is not '
                   'the code alone.')
        around = conditions_changed(run, m, prev)
        if around:
            vs += f' <b>Also different</b>: {esc(around)}.'
    return f"""<div class="mpart">
  <h4>Startup and single-user speed</h4>
  <p class="msub">One row per engine. Startup in seconds (warm in ms); one user at {shape or 'the base shape'}.{vs}
  What the columns mean: startup {STARTUP_INFO} one user {TIMING_INFO} peak RSS {RSS_INFO}</p>
<div class="mtable"><table>
  <thead>
    <tr><th rowspan="2" scope="col" class="first">engine</th>
        <th colspan="4" scope="colgroup">startup</th>
        <th colspan="3" scope="colgroup">one user{', ' + shape if shape else ''}</th>
        <th rowspan="2" scope="col">peak RSS<br><span class="dim">MiB</span></th></tr>
    <tr><th scope="col">frozen<br><span class="dim">s</span></th><th scope="col">cold<br><span class="dim">s</span></th>
        <th scope="col">1st req<br><span class="dim">s</span></th><th scope="col">warm<br><span class="dim">ms</span></th>
        <th scope="col">TTFT<br><span class="dim">ms</span></th><th scope="col">TPOT<br><span class="dim">ms</span></th>
        <th scope="col">tok/s</th></tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table></div>
</div>"""


def line_svg(series, xs, cid, ylabel, nd, title, head, xlabel, notes=None):
    """One small line chart, the unit of every small multiple: x is the axis
    being swept (users, or prompt tokens), one line per series. A point that was
    not measured (None) leaves a gap in its line rather than a bridge. Its
    legend sits above it, so it carries no end labels and stays narrow.
    `notes` maps (series label, x) to a suffix for that point's tooltip."""
    notes = notes or {}
    W, H, L, R, T, B = 300, 190, 44, 14, 10, 36
    pw, ph = W - L - R, H - T - B
    vals = [v for _, _, pts in series for _, v in pts if v is not None]
    ymax = nice_max(max(vals))
    xpos = {x: L + (pw * i / (len(xs) - 1) if len(xs) > 1 else pw / 2) for i, x in enumerate(xs)}

    def ypos(v):
        return T + ph - ph * v / ymax

    def short(x):                         # 131072 -> 128k, so long prompt lengths fit
        return f"{x // 1024}k" if x >= 8192 and x % 1024 == 0 else f"{x:,}"

    out = [f'<svg viewBox="0 0 {W} {H}" class="mchart" role="img" '
           f'aria-labelledby="{cid}-t"><title id="{cid}-t">{esc(title)}</title>']
    for i in range(5):
        v = ymax * i / 4
        y = ypos(v)
        out.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for x in xs:
        out.append(f'<text class="tick" x="{xpos[x]:.1f}" y="{T + ph + 16}" text-anchor="middle">{short(x)}</text>')
    out.append(f'<text class="axis" x="{L + pw / 2:.1f}" y="{H - 3}" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text class="axis" x="11" y="{T + ph / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 11 {T + ph / 2:.1f})">{esc(ylabel)}</text>')
    out.append(f'<line class="xhair" x1="0" x2="0" y1="{T}" y2="{T + ph}" style="display:none"/>')
    for label, cls, pts in series:
        d, pen = [], "M"
        for x, v in pts:
            if v is None:
                pen = "M"
                continue
            d.append(f"{pen}{xpos[x]:.1f},{ypos(v):.1f}")
            pen = "L"
        out.append(f'<path class="ln {cls}" d="{" ".join(d)}"/>')
        out += [f'<circle class="dot {cls}" cx="{xpos[x]:.1f}" cy="{ypos(v):.1f}" r="4"/>'
                for x, v in pts if v is not None]
    # One hit band per x, wider than any mark: the crosshair finds the x and
    # the tooltip lists every series there.
    by_x = [(label, cls, dict(pts)) for label, cls, pts in series]
    for i, x in enumerate(xs):
        left = (xpos[xs[i - 1]] + xpos[x]) / 2 if i else L
        right = (xpos[x] + xpos[xs[i + 1]]) / 2 if i + 1 < len(xs) else L + pw
        rows = []
        for label, cls, at in by_x:
            # A point present but None was not measured (no stream); absent is "—".
            value = NO_STREAM if x in at and at[x] is None else fmt(at.get(x), nd)
            note = notes.get((label, x))
            rows.append([value + (PARTIAL if note else ""), label + (f" ({note})" if note else ""), cls])
        out.append(f'<rect class="hit" x="{left:.1f}" y="{T}" width="{right - left:.1f}" height="{ph}" '
                   f'tabindex="0" data-x="{xpos[x]:.1f}" data-head="{short(x)} {esc(head)}" '
                   f'data-tip="{tip(rows)}"><title>{short(x)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


# The two sweeps a model gets small multiples for: (axis, section title, x-axis
# label, tooltip unit, caption, the metrics as (metric, title, unit note, y label,
# digits)). Two measures each, so two rows of charts rather than two y axes.
SWEEPS = {
    "conc": ("As users are added", "offered users", "users",
             "A rising line in the lower row means each user's answer slows down as more share the engine.",
             [("output_throughput", "Throughput", "tok/s, higher is better", "tok/s", 1),
              ("median_tpot_ms", "Time per output token", "ms, lower is better", "ms", 1)]),
    "input": ("As prompts get longer", "prompt tokens", "token prompt",
              "Memory is the whole server's footprint at its peak during the cell, Metal buffers "
              "included; scratchy reserves its KV cache up front, so its line starts high and stays flat.",
              [("median_ttft_ms", "Time to first token", "ms, lower is better", "ms", 0),
               ("server_mem_peak_mib", "Server memory", "MiB, peak", "MiB", 0)]),
}


def sweep_charts(m, mid, axis, base, run, prev=None):
    """One small multiple per comparison engine, scratchy against it alone: two
    colours per chart however many engines ran. The previous run's scratchy line
    sits behind in grey."""
    title, xlabel, head, caption, metrics = SWEEPS[axis]
    engines = [(k, lab, {c["rung"]: c for c in cells if c.get("axis") == axis})
               for k, lab, _l, cells in engines_in(m)]
    engines = [e for e in engines if e[2]]
    if not engines or engines[0][0] != "scratchy":
        return ""
    mine = engines[0][2]
    rivals = engines[1:] or [(None, None, {})]
    xs = sorted({x for _k, _l, cs in engines for x in cs})
    pcs = {c["rung"]: c for c in (prev[1].get("scaling") if prev else None) or [] if c.get("axis") == axis}
    plabel = ""
    if prev and pcs:
        plabel = f"scratchy, previous ({prev[0]['generated_utc'][5:10]}" + (
            ", different settings)" if settings_changed(run, prev[0]) else ")")

    def pts(cs, metric):
        return [(x, timing(cs[x], metric)) for x in xs if x in cs]

    rows = []
    for metric, mtitle, unit, ylabel, nd in metrics:
        panes = []
        for key, label, theirs in rivals:
            series = []
            if pcs:
                series.append((plabel, "prev", pts(pcs, metric)))
            series.append(("scratchy", "s1", pts(mine, metric)))
            if key:
                series.append((label, "s2", pts(theirs, metric)))
            series = [x for x in series if any(v is not None for _, v in x[2])]
            if not any(lbl == "scratchy" for lbl, _c, _p in series):
                continue
            marks = {(lab, x): untimed_note(cs[x]) for lab, cs in (("scratchy", mine), (label, theirs))
                     if lab for x in cs if partial(cs[x], metric)}
            gone = key and label not in {lbl for lbl, _c, _p in series}
            name = (f"vs {esc(label)}" + (f' <span class="dim">({NO_STREAM})</span>' if gone else "")
                    if key else "scratchy")
            svg = line_svg(series, xs, f"{mid}-{axis}-{metric}-{key or 'self'}", ylabel, nd,
                           f"{mtitle} against {xlabel}, scratchy and {label or 'nothing else'}",
                           f"{head} · {mtitle}", xlabel, marks)
            panes.append(f'<div class="chartpane mini"><div class="minititle">{name}</div>{svg}</div>')
        if panes:
            # One legend per row: the same two or three lines in every chart.
            keys = [("s1", "scratchy")] + ([("s2", "the engine named above each chart")] if rivals[0][0] else [])
            if pcs:
                keys.append(("prev", plabel))
            legend = "".join(f'<span><span class="lkey {c}"></span>{esc(t)}</span>' for c, t in keys)
            rows.append(f'<div class="heatttl"><b>{esc(mtitle)}</b> <span class="dim">({esc(unit)})</span></div>'
                        f'<div class="legend">{legend}</div><div class="chartrow">{"".join(panes)}</div>')
    if not rows:
        return ""
    # The table view: every engine, every metric, every x.
    head_row = "".join(f'<th scope="col">{x:,}</th>' for x in xs)
    trs = []
    for _key, label, cs in engines:
        for metric, mtitle, _u, ylabel, nd in metrics + ([("median_ttft_ms", "TTFT", "", "ms", 0)] if axis == "conc" else []):
            trs.append(f'<tr><th scope="row">{esc(label)} · {esc(mtitle)} {esc(ylabel)}</th>'
                       + "".join(f"<td>{cfmt(cs.get(x), metric, nd)}</td>" for x in xs) + "</tr>")
    shape = (f"{', '.join(map(str, xs[:-1]))} and {xs[-1]} users at once {USERS_INFO}; "
             f"{esc(base.get('input', '?'))}-token prompts, {esc(base.get('output', '?'))}-token answers"
             if axis == "conc" else
             f"{esc(base.get('conc', '?'))} users at once, {esc(base.get('output', '?'))}-token answers")
    return f"""<figure class="mfig">
  <figcaption><h4>{esc(title)}</h4>
    <p class="msub">{shape}. One small chart per engine: scratchy in blue against that engine in orange. {esc(caption)}</p></figcaption>
  {"".join(rows)}
  {fold("Table view", f'<div class="mtable"><table><thead><tr><th scope="col">{esc(xlabel)}</th>{head_row}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')}
</figure>"""


# ColorBrewer RdBu, 7 classes: diverging, colour-blind safe, stepped so "a bit
# faster" and "much faster" read differently. Bounds are symmetric in log space
# (x0.8 is as far from x1 as x1.25) and the middle class is the noise band.
# The colours themselves live in styles.css as .hm0 to .hm6.
HEAT_BOUNDS = [0.5, 0.8, 1 - NOISE, 1 + NOISE, 1.25, 2]
HEAT_LABELS = ["much slower (x0.5 or less)", "slower", "a bit slower", "about the same",
               "a bit faster", "faster", "much faster (x2 or more)"]


def heat_legend():
    """One swatch per class, slowest to fastest."""
    return "".join(f'<span><i class="sw hm{i}"></i>{esc(label)}</span>' for i, label in enumerate(HEAT_LABELS))


def heat_class(r):
    """The RdBu class for a "how many times faster is scratchy" ratio: hm0 (much
    slower) to hm6 (much faster), hm3 inside the noise band; "" with no ratio."""
    if r is None:
        return ""
    return f"hm{sum(r >= b for b in HEAT_BOUNDS)}"


def faster_tput(s, o):
    """How many times faster scratchy's throughput is (higher is better)."""
    return s / o if s and o else None


def faster_ttft(s, o):
    """How many times faster scratchy's time to first token is (lower is better)."""
    return o / s if s and o else None


GRID_METRICS = [("output_throughput", "throughput", faster_tput),
                ("median_ttft_ms", "time to first token", faster_ttft)]


def grid_setup(m, run):
    """A model's prompt-by-answer grid: (inputs, outputs, cells per engine key,
    the comparison engines in page order, the headline one), or None when
    scratchy has no grid. The headline engine follows RIVAL_PREFERENCE."""
    sc = run["config"].get("scaling") or {}
    ins, outs = sc.get("grid_input") or [], sc.get("grid_output") or []
    have = {k: {c["rung"]: c for c in cells if c.get("axis") == "grid"} for k, _l, _ld, cells in engines_in(m)}
    have = {k: v for k, v in have.items() if v}
    if not have.get("scratchy") or not ins or not outs:
        return None
    rivals = [k for k, _l in ENGINE_ORDER if k != "scratchy" and k in have]
    headline = next((k for k in RIVAL_PREFERENCE if k in have), None)
    return ins, outs, have, rivals, headline


def grid_maps(m, run):
    """The prompt-by-answer grid, dense: per metric one table, scratchy's own
    values on top, then one row per engine with each cell coloured by how many
    times faster scratchy is than it. Every engine in one view, no picker."""
    setup = grid_setup(m, run)
    if setup is None:
        return ""
    ins, outs, have, rivals, _headline = setup
    conc = base_of(run, m).get("conc", "?")

    def table(metric, title, faster):
        nd = 1 if metric == "output_throughput" else 0
        unit = "tok/s" if metric == "output_throughput" else "ms"
        groups = "".join(f'<th scope="colgroup" colspan="{len(outs)}" class="grp">{i:,}-token prompt</th>' for i in ins)
        heads = "".join(f'<th scope="col" class="{"grp0" if j == 0 else ""}">{o}</th>' for _ in ins for j, o in enumerate(outs))
        mine = have["scratchy"]
        trs = ['<tr><th scope="row">scratchy <span class="dim">' + unit + '</span></th>'
               + "".join(f'<td class="own{" grp0" if j == 0 else ""}">{cfmt(mine.get(f"{i}x{o}"), metric, nd)}</td>'
                         for i in ins for j, o in enumerate(outs)) + "</tr>"]
        for k in rivals:
            tds = []
            for i in ins:
                for j, o in enumerate(outs):
                    rung = f"{i}x{o}"
                    cs, cr = mine.get(rung), have[k].get(rung)
                    r = faster(timing(cs, metric), timing(cr, metric))
                    mark = PARTIAL if partial(cs, metric) or partial(cr, metric) else ""
                    text = (f"{r:.2f}{mark}" if r is not None
                            else NO_STREAM if cr and timing(cr, metric) is None else "—")
                    rows = [[cfmt(cs, metric, nd), "scratchy", "s1"], [cfmt(cr, metric, nd), LABEL[k], "s2"]]
                    tds.append(f'<td class="{heat_class(r)}{" grp0" if j == 0 else ""}" tabindex="0" '
                               f'data-head="{i} in × {o} out · {esc(title)}" data-tip="{tip(rows)}">{text}</td>')
            trs.append(f'<tr><th scope="row">vs {esc(LABEL[k])}</th>{"".join(tds)}</tr>')
        return f"""<div class="heat dheat">
    <div class="heatttl"><b>{esc(title)}</b></div>
    <div class="mtable"><table><thead><tr><th scope="col" rowspan="2"><span class="dim">answer tokens →</span></th>{groups}</tr>
    <tr>{heads}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>
  </div>"""

    note = (f"{esc(conc)} users at once. Top row: scratchy's own values. Each row below: how many times "
            "faster scratchy is than that engine, ×1.20 is 20% faster, below 1 is slower. Hover for both values."
            if rivals else f"{esc(conc)} users at once; scratchy's own values, with no other engine to compare against.")
    scale = (f'<div class="scale"><span class="scalekey">scratchy is:</span>{heat_legend()}</div>' if rivals else "")
    return f"""<figure class="mfig">
  <figcaption><h4>Prompt size × answer size</h4>
    <p class="msub">{note}</p></figcaption>
  {"".join(table(metric, title, faster) for metric, title, faster in GRID_METRICS)}
  {scale}
</figure>"""


def history(entries):
    if len(entries) < 2:
        return ""
    rows = []
    for run, m in reversed(entries):
        sha = str(run["repo"]["sha"])[:8]
        ladder, one = m.get("cache_ladder"), cell(m.get("scaling"), "conc", 1)
        top = max((c["rung"] for c in m.get("scaling") or [] if c.get("axis") == "conc"), default=None)
        many = cell(m.get("scaling"), "conc", top) if top else None
        if served(m):
            vals = [fmt(med(ladder, "cold", "t_ready_s"), 2), fmt(med(ladder, "cold", "ttft_from_send_s"), 2),
                    fmt(med(ladder, "warm", "ttft_from_send_s"), 0, 1000),
                    fmt(timing(one, "output_throughput"), 1),
                    fmt(timing(many, "output_throughput"), 1), cfmt(many, "median_tpot_ms", 1)]
            tds = "".join(f"<td>{v}</td>" for v in vals)
        else:
            tds = '<td colspan="6" class="gap">no numbers</td>'
        rows.append(f'<tr><th scope="row">{run["generated_utc"][:10]}</th>'
                    f'<td><a href="{REPO}/commit/{esc(run["repo"]["sha"])}"><code>{sha}</code></a>'
                    '</td>'
                    f'<td>{esc(m.get("quant") or "")}</td>{tds}</tr>')
    return fold(f"Runs of this model ({len(entries)})", f"""<div class="mtable"><table>
  <thead><tr><th scope="col">run</th><th scope="col">scratchy</th><th scope="col">quant</th>
  <th scope="col">cold s</th><th scope="col">1st req s</th><th scope="col">warm ms</th><th scope="col">tok/s, 1 user</th>
  <th scope="col">tok/s, most users</th><th scope="col">TPOT ms, most users</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>""")


def engine_versions(run):
    """The comparison engines' releases as the runner recorded them: every
    engine from config.engines, else the mlx-lm/ollama pair older runs kept."""
    c = run["config"]
    if c.get("engines"):
        parts = [f"{e['label']} {e['version']}" for e in c["engines"]
                 if e.get("key") != "scratchy" and e.get("installed") and e.get("version")]
    else:
        v = c.get("engine_versions") or {}
        parts = [f"{LABEL.get(k, k)} {v[k]}" for k in ("mlx_lm", "ollama") if v.get(k)]
    return " · ".join(parts) or "not recorded"


def pinned(run):
    """Whether the run was an agreed, fully pinned one (runs before pins: not recorded)."""
    c = run["config"]
    if "pinned" not in c:
        return "not recorded"
    return "yes" if c["pinned"] else "no: " + "; ".join(c.get("pin_problems") or [])


def parity(m):
    """The parity gate's verdict for the model line: scratchy and mlx-lm must
    give the same greedy answers before anything is timed. Runs from before
    the gate say nothing."""
    if "parity_mlx_lm" not in m:
        return ""
    verdict = {True: "matches mlx-lm", False: "does not match mlx-lm"}.get(m["parity_mlx_lm"],
                                                                          "not checked (no mlx-lm)")
    return " · output " + esc(verdict) + " " + PARITY_INFO


PARITY_INFO = info(
    "Before anything is timed, scratchy and mlx-lm answer the same short prompts with greedy "
    "decoding, and the answers must match exactly. A mismatch points at a wrong load path "
    "(quant preset, group size, dequant), so that model is not timed at all.")


def runs_table(runs):
    rows = []
    for run in reversed(runs):
        r, c = run["repo"], run["config"]
        prime = c.get("cold_priming_launches")
        rows.append(
            f'<tr><th scope="row">{esc(run["generated_utc"][:16].replace("T", " "))} UTC</th>'
            f'<td><a href="{REPO}/commit/{esc(r["sha"])}"><code>{esc(str(r["sha"])[:8])}</code></a>'
            '</td>'
            f'<td>{esc(", ".join(c.get("scenarios") or []))}</td>'
            f'<td>{esc(prime) if prime is not None else "not recorded"}</td>'
            f'<td>{esc(settings(run)["KV cache"])}</td>'
            f'<td><code>{esc(settings(run)["serve flags"])}</code></td>'
            f'<td>{esc(engine_versions(run))}</td>'
            f'<td>{esc(pinned(run))}</td>'
            f'<td><a href="data/metal/{esc(run["_file"].name)}">json</a></td></tr>')
    return fold(f"Runs on this machine ({len(runs)})", f"""<div class="mtable"><table>
  <thead><tr><th scope="col">when</th><th scope="col">scratchy</th><th scope="col">steps</th>
  <th scope="col">cold priming launches</th><th scope="col">scratchy KV cache</th><th scope="col">scratchy serve flags</th><th scope="col">engine versions</th><th scope="col">pinned</th><th scope="col">data</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>""")


def current_models(run):
    """The models the runner measured as current when this run was made: its
    full set (current_models), or the default list the first runs to record
    one kept; none for runs older than that."""
    return run["config"].get("current_models") or run["config"].get("default_models")


def index(runs):
    """Per machine, oldest run first: (runs, newest entry per current model,
    every entry per model). Newer runs win, so a model shows its newest numbers.
    A model is retired once the newest run that recorded the runner's current
    models leaves it out and ran it no later, so a --models subset hides
    nothing; runs before that record keep every model current."""
    machines = {}
    for run in sorted(runs, key=lambda r: r["generated_utc"]):
        rs, latest, seen = machines.setdefault(run["machine"]["chip"], ([], {}, {}))
        rs.append(run)
        for m in run["models"]:
            latest[m["stem"]] = (run, m)
            seen.setdefault(m["stem"], []).append((run, m))
    for rs, latest, _seen in machines.values():
        ref = next((r for r in reversed(rs) if current_models(r)), None)
        if ref:
            keep = set(current_models(ref))
            for stem in [s for s, (r, _m) in latest.items()
                         if s not in keep and r["generated_utc"] <= ref["generated_utc"]]:
                del latest[stem]
    return dict(sorted(machines.items()))


def glance(machines):
    """Every model's prompt-by-answer grid on every machine as one small
    multiple per metric: the same colours as the full grids, no numbers in the
    squares, each linking to its model's full section."""
    stems = []
    for _rs, latest, _seen in machines.values():
        stems += [stem for stem in latest if stem not in stems]
    if not stems:
        return ""

    def mini(chip, stem, metric, faster):
        entry = machines[chip][1].get(stem)
        if entry is None:
            return '<td class="gnone">not run</td>'
        run, m = entry
        setup = grid_setup(m, run)
        if setup is None:
            return '<td class="gnone">no grid</td>'
        ins, outs, have, _rivals, rival = setup
        if rival is None:
            return '<td class="gnone">nothing to compare</td>'
        squares, ratios = [], []
        for i in ins:
            for o in outs:
                rung = f"{i}x{o}"
                cs, cr = have["scratchy"].get(rung), have[rival].get(rung)
                r = faster(timing(cs, metric), timing(cr, metric))
                mark = PARTIAL if partial(cs, metric) or partial(cr, metric) else ""
                if r is not None:
                    ratios.append(r)
                value = f"×{r:.2f}{mark}" if r is not None else "no comparison"
                # A square with nothing to compare is hollow, so it never reads
                # as "about the same".
                squares.append(f'<span class="gsq {heat_class(r) or "gnil"}" data-head="{esc(chip)} · {esc(stem)} · {i} in × {o} out" '
                               f'data-tip="{tip([[value, "vs " + LABEL[rival], "s1"]])}"></span>')
        span = f"×{min(ratios):.2f} to ×{max(ratios):.2f}" if ratios else "no ratios"
        return (f'<td><a class="gmini" href="#{slug(chip)}-{slug(stem)}" '
                f'aria-label="{esc(stem)} on {esc(chip)}: scratchy {span} against {esc(LABEL[rival])}; open the full grid" '
                f'style="grid-template-columns: repeat({len(outs)}, 1fr)">{"".join(squares)}</a>'
                f'<span class="grange">{span} vs {esc(LABEL[rival])}</span></td>')

    tables = []
    for metric, title, faster in GRID_METRICS:
        head = "".join(f'<th scope="col">{esc(stem)}</th>' for stem in stems)
        rows = "".join(f'<tr><th scope="row">{esc(chip)}</th>'
                       + "".join(mini(chip, stem, metric, faster) for stem in stems) + "</tr>"
                       for chip in machines)
        tables.append(f'<div class="heat"><div class="heatttl">{esc(title)}</div>'
                      f'<div class="mtable"><table class="gtable"><thead><tr><th scope="col"></th>{head}</tr></thead>'
                      f'<tbody>{rows}</tbody></table></div></div>')
    return f"""<section class="msection" id="glance">
  <h2>At a glance</h2>
  <p class="msub">Every model's prompt size × answer size grid on every machine. Each square is
  one cell of the full grid (rows: prompt, short to long; columns: answer, short to long),
  coloured by how many times faster scratchy is than one engine: oMLX with TurboQuant where the
  run has it (TurboQuant against TurboQuant), else oMLX, else mlx-lm; each grid names its engine.
  Hover a square for its ratio; click a grid for every engine.</p>
  <div class="heatrow">{''.join(tables)}</div>
  <div class="scale"><span class="scalekey">scratchy is:</span>{heat_legend()}<span><i class="sw nil"></i>no comparison</span></div>
</section>"""


def machine_section(chip, runs, latest, seen):
    mslug = slug(chip)
    mc = runs[-1]["machine"]
    cores = f'{mc["cores_total"]} cores'
    if mc.get("cores_performance"):
        cores += f' ({mc["cores_performance"]}P + {mc.get("cores_efficiency", 0)}E)'
    parts = [f"""<section class="msection" id="{mslug}">
  <h2>{esc(chip)}</h2>
  <p class="mmeta">{cores} · {esc(mc["memory_gb"])} GB unified memory · macOS {esc(mc["macos"])}</p>
  {runs_table(runs)}"""]
    skipped = [x for x in runs[-1].get("skipped_models") or [] if x.get("stem") not in latest]
    if skipped:
        parts.append('  <p class="msub">Not run on this machine: '
                     + "; ".join(f'{esc(x["stem"])} ({esc(x.get("reason") or "skipped")})' for x in skipped)
                     + ".</p>")
    for stem, (run, m) in latest.items():
        # Before/after: the newest earlier run of this model, here, that scratchy served.
        prev = next(((r, pm) for r, pm in reversed(seen[stem][:-1]) if served(pm)), None)
        mid = f"{mslug}-{slug(stem)}"
        f = m.get("footprint") or {}
        build = ""
        if f.get("build_seconds") is not None:
            build = f' · scratchy build {esc(f["build_seconds"])} s'
            if f.get("binary_bytes"):
                build += f', {round(f["binary_bytes"] / 1048576)} MiB binary'
        rev = (m.get("model_files") or {}).get("revision")
        files = (f' · files <a href="https://huggingface.co/{esc(m["model_id"])}/tree/{esc(rev)}">'
                 f'<code>{esc(rev[:8])}</code></a>') if rev else ""
        base = base_of(run, m)
        parts.append(f"""  <article class="mmodel cds--tile" id="{mid}">
    <h3>{esc(stem)} <span class="onmachine">on {esc(chip)}</span></h3>
    <p class="mmeta"><a href="https://huggingface.co/{esc(m["model_id"])}">{esc(m["model_id"])}</a>
      · {esc(m.get("quant") or "default")}{files}{build}{parity(m)}
      · run {esc(run["generated_utc"][:10])}, <code>{esc(str(run["repo"]["sha"])[:8])}</code></p>
    {summary_table(m, run, prev)}
    {sweep_charts(m, mid, "conc", base, run, prev)}
    {grid_maps(m, run)}
    {sweep_charts(m, mid, "input", base, run, prev)}
    {history(seen[stem])}
  </article>""")
    parts.append("</section>")
    return "\n".join(parts)


def page(header, side_nav, data_dir):
    runs = load(data_dir)
    machines = index(runs)
    if machines:
        body = glance(machines) + "\n" + "\n".join(
            machine_section(chip, *entry) for chip, entry in machines.items())
        sections = [("glance", "At a glance")] + [(slug(chip), chip) for chip in machines]
    else:
        body = ('<p class="empty">No runs published yet. Run <code>scripts/bench_metal_matrix.sh</code> '
                'and copy its JSON into <code>site/data/metal/</code>.</p>')
        sections = []
    fill = {"header": header, "sidenav": side_nav("metal.html", sections), "body": body,
            "count": str(len(runs)), "machines": str(len(machines)), "repo": REPO,
            "about": ABOUT}
    # One pass, so text a placeholder inserts is never itself rewritten.
    out = re.sub(r"\{(" + "|".join(fill) + r")\}", lambda mo: fill[mo.group(1)], PAGE)
    return out, runs, len(machines)


def build(out_path, header, side_nav, data_dir=DATA):
    out, runs, n = page(header, side_nav, data_dir)
    out_path = Path(out_path)
    out_path.write_text(out)
    dest = out_path.parent / "data" / "metal"
    dest.mkdir(parents=True, exist_ok=True)
    for run in runs:
        shutil.copy(run["_file"], dest / run["_file"].name)
    print(f"metal page: {len(runs)} runs on {n} machines -> {out_path}")


def main():
    sys.path.insert(0, str(HERE))
    from build import header_html, perf_side_nav  # the one definition of the site chrome

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_site/metal.html"
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA
    build(out, header_html("", active="metal.html"), perf_side_nav, data)


ABOUT = info(
    "Prompts are made up, a fixed size, and unique per request so no cache can answer "
    "them, with greedy decoding (temperature 0) on every engine. oMLX, mlx-lm and "
    "vllm-metal run the same MLX checkpoint as scratchy; ollama runs its MLX engine; "
    "llama.cpp and mistral.rs run the same Q4_K_M GGUF, named on its row. oMLX runs twice: "
    "as installed (FP16 KV) and with its TurboQuant KV, like scratchy's default. "
    "scratchy's build time is shown per model and is never part of startup. The (i) "
    "buttons next to a column explain it.", "About these numbers")

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Metal performance — scratchy</title>
<meta name="description" content="scratchy against mlx-lm and ollama on Apple silicon: startup, single-user speed, and scaling, per machine.">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@carbon/styles@1/css/styles.min.css">
<link rel="stylesheet" href="styles.css">
<script type="module" src="https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest/ui-shell.min.js"></script>
<script type="module" src="https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest/toggle-tip.min.js"></script>
<script type="module" src="https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest/accordion.min.js"></script>
<script>
// Always Carbon's light theme: the heat-map steps and series colours are
// chosen and checked against it, whatever the reader's system setting.
document.documentElement.classList.add('cds--white');
</script>
</head>
<body>

{header}

{sidenav}

<div class="docs-layout">
<main class="docs-content metalpage">
  <div class="hero-body">
    <h1>Metal performance</h1>
    <p class="lede">scratchy against mlx-lm and ollama on Apple silicon: how long each takes
    to start, how fast it answers one user, and how it holds up as prompts, answers and users
    grow. {count} runs on {machines} machines, every number measured by
    <a href="{repo}/blob/main/scripts/bench_metal_matrix.sh"><code>scripts/bench_metal_matrix.sh</code></a>.</p>
    <div class="mabout">{about}</div>
  </div>


{body}
</main>
</div>

<div id="mtip" class="mtip" role="tooltip" hidden></div>
<script>
// Hover and keyboard focus show the same readout. Values come from data-tip
// and are written with textContent only.
(function () {
  var tipEl = document.getElementById('mtip');
  function show(el) {
    var rows = JSON.parse(el.getAttribute('data-tip') || '[]');
    tipEl.textContent = '';
    var h = document.createElement('div');
    h.className = 'mtiphead';
    h.textContent = el.getAttribute('data-head') || '';
    tipEl.appendChild(h);
    rows.forEach(function (r) {
      var row = document.createElement('div');
      var key = document.createElement('span');
      key.className = 'lkey ' + r[2];
      var v = document.createElement('b');
      v.textContent = r[0];
      var l = document.createElement('span');
      l.className = 'dim';
      l.textContent = ' ' + r[1];
      row.append(key, v, l);
      tipEl.appendChild(row);
    });
    tipEl.hidden = false;
    var b = el.getBoundingClientRect();
    tipEl.style.left = (window.scrollX + Math.min(window.innerWidth - tipEl.offsetWidth - 8, b.left + b.width / 2 + 12)) + 'px';
    tipEl.style.top = (b.top + window.scrollY + 8) + 'px';
    var x = el.getAttribute('data-x');
    var hair = x && el.ownerSVGElement && el.ownerSVGElement.querySelector('.xhair');
    if (hair) { hair.setAttribute('x1', x); hair.setAttribute('x2', x); hair.style.display = ''; }
  }
  function hide(el) {
    tipEl.hidden = true;
    var hair = el.ownerSVGElement && el.ownerSVGElement.querySelector('.xhair');
    if (hair) hair.style.display = 'none';
  }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointerenter', function () { show(el); });
    el.addEventListener('pointerleave', function () { hide(el); });
    el.addEventListener('focus', function () { show(el); });
    el.addEventListener('blur', function () { hide(el); });
  });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

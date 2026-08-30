"""Generate the reproducible environment file and the Wadati QC notebook.

Run from the repository root:

    python -m app.notebook.build_notebook

Writes ``notebooks/environment.yml`` and ``notebooks/wadati_qc.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT_DIR = Path("notebooks")
NOTEBOOK_NAME = "wadati_qc.ipynb"

# Pure-Python modules copied next to the notebook so the notebook exercises the
# same loader code the app ships (and the code stays unit-testable).
MODULE_MAP = {
    "ingest.py": "wadati_ingest.py",
    "samples.py": "wadati_samples.py",
    "discovery.py": "wadati_discovery.py",
    "subset_search.py": "wadati_subset.py",
    "subset_examples.py": "wadati_subset_examples.py",
    "dashboard.py": "wadati_dashboard.py",
}

THEME_CELL = """\
# --- Field-notebook theme (warm mineral paper / slate / seismic accents) ---
THEME = {
    "paper": "#F6F1E7",      # warm mineral paper
    "paper_deep": "#EDE5D6",  # sub-panel wash
    "rule": "#CBBFA6",        # thin geological rule line
    "ink": "#2B2F33",         # charcoal editorial ink
    "slate": "#5C666F",       # secondary slate text
    "teal": "#0E6B6B",        # retained / pass
    "red": "#A6321F",         # rejected / outlier
}

mpl.rcParams.update({
    "figure.facecolor": THEME["paper"],
    "axes.facecolor": THEME["paper"],
    "axes.edgecolor": THEME["rule"],
    "axes.labelcolor": THEME["ink"],
    "axes.titlecolor": THEME["ink"],
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": THEME["rule"],
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
    "text.color": THEME["ink"],
    "xtick.color": THEME["slate"],
    "ytick.color": THEME["slate"],
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "savefig.facecolor": THEME["paper"],
})

pn.extension(design="native")
pn.config.raw_css = [
    f"body, .bk-root {{ background: {THEME['paper']}; color: {THEME['ink']}; }}"
]
print("Theme registered:", ", ".join(THEME))
"""

SAMPLE_CELL = """\
# --- Sample phase-pick table (one event, 12 stations) ---
# ts_p : P travel time (s) from origin;  s_minus_p : S-P interval (s)
# A homogeneous medium gives S-P = (Vp/Vs - 1) * ts_p  ->  slope = Vp/Vs - 1.
rng = np.random.default_rng(20240517)

stations = ["BRG", "MOX", "CLL", "TANN", "WERD", "PLN",
            "SCHF", "GUNZ", "ROHR", "NEUB", "LAUE", "HAIN"]
ts_p = np.array([2.6, 4.1, 5.5, 6.9, 8.2, 9.6, 11.1, 12.4, 14.0, 15.8, 17.2, 18.9])

true_vpvs = 1.73
s_minus_p = (true_vpvs - 1.0) * ts_p + rng.normal(0.0, 0.09, ts_p.size)

# Inject two deliberate mis-picks so the QC states are visible.
s_minus_p[4] += 1.9    # WERD: late S pick
s_minus_p[9] -= 1.6    # NEUB: early S pick

picks = pd.DataFrame({
    "station": stations,
    "ts_p": ts_p,
    "s_minus_p": s_minus_p,
})
picks.head(12)
"""

FIT_CELL = '''\
def wadati_fit(frame: pd.DataFrame) -> dict:
    """Least-squares Wadati regression through the pick table."""
    fit = linregress(frame["ts_p"].to_numpy(), frame["s_minus_p"].to_numpy())
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "vp_vs": float(fit.slope) + 1.0,
        "stderr": float(fit.stderr),
    }


def flag_outliers(frame: pd.DataFrame, fit: dict, n_sigma: float = 2.0) -> pd.Series:
    """Mark picks whose residual exceeds n_sigma of the residual spread."""
    predicted = fit["intercept"] + fit["slope"] * frame["ts_p"]
    residual = frame["s_minus_p"] - predicted
    return (residual.abs() > n_sigma * residual.std(ddof=1))


raw_fit = wadati_fit(picks)
picks["rejected"] = flag_outliers(picks, raw_fit)
kept = picks.loc[~picks["rejected"]]
clean_fit = wadati_fit(kept)

print(f"all picks   n={len(picks):2d}  r={raw_fit['r']:.4f}  Vp/Vs={raw_fit['vp_vs']:.3f}")
print(f"retained    n={len(kept):2d}  r={clean_fit['r']:.4f}  Vp/Vs={clean_fit['vp_vs']:.3f}")
print("rejected    ", ", ".join(picks.loc[picks['rejected'], 'station']) or "none")
'''

PLOT_CELL = """\
def wadati_plot(frame: pd.DataFrame, fit: dict, title: str = "Wadati diagram") -> Figure:
    fig = Figure(figsize=(6.4, 4.0), dpi=110)
    ax = fig.subplots()

    keep = frame.loc[~frame["rejected"]]
    drop = frame.loc[frame["rejected"]]

    grid = np.linspace(0.0, frame["ts_p"].max() * 1.08, 50)
    ax.plot(grid, fit["intercept"] + fit["slope"] * grid,
            color=THEME["teal"], lw=1.4, zorder=2,
            label=f"fit  Vp/Vs = {fit['vp_vs']:.3f}   r = {fit['r']:.4f}")

    ax.scatter(keep["ts_p"], keep["s_minus_p"], s=42, zorder=3,
               facecolor=THEME["teal"], edgecolor=THEME["paper"], lw=0.8,
               label=f"retained (n={len(keep)})")
    ax.scatter(drop["ts_p"], drop["s_minus_p"], s=52, zorder=3, marker="X",
               facecolor=THEME["red"], edgecolor=THEME["paper"], lw=0.8,
               label=f"rejected (n={len(drop)})")

    for _, row in frame.iterrows():
        ax.annotate(row["station"], (row["ts_p"], row["s_minus_p"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                    color=THEME["red"] if row["rejected"] else THEME["slate"])

    ax.set_xlabel("P travel time  $t_P$  (s)")
    ax.set_ylabel("$t_S - t_P$  (s)")
    ax.set_title(title, loc="left", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


centerpiece = pn.Column(
    pn.pane.Markdown(
        f"### Event 2024-05-17T04:12:09Z — single-event Wadati QC\\n"
        f"Retained **{int((~picks['rejected']).sum())}** of **{len(picks)}** station pairs.",
        styles={"color": THEME["ink"]},
    ),
    pn.pane.Matplotlib(wadati_plot(picks, clean_fit), tight=True, format="svg"),
    pn.pane.Markdown(
        f"`Vp/Vs = {clean_fit['vp_vs']:.3f} ± {clean_fit['stderr']:.3f}`  ·  "
        f"`r = {clean_fit['r']:.4f}`  ·  rejected: "
        f"{', '.join(picks.loc[picks['rejected'], 'station']) or 'none'}",
        styles={"color": THEME["slate"]},
    ),
    styles={"background": THEME["paper"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
centerpiece
"""

VERIFY_CELL = """\
# --- Environment self-check (safe to re-run; no model weights downloaded) ---
import importlib

for name in ["numpy", "scipy", "pandas", "matplotlib", "obspy", "panel", "nbformat"]:
    mod = importlib.import_module(name)
    print(f"{name:12s} {getattr(mod, '__version__', 'n/a')}")

try:
    import torch
    print(f"{'torch':12s} {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
    import seisbench.models as sbm
    print(f"{'seisbench':12s} PhaseNet weights: {len(sbm.PhaseNet.list_pretrained())} available")
except Exception as exc:  # pragma: no cover - optional in a light install
    logging_msg = f"SeisBench / torch not ready yet: {exc}"
    print(logging_msg)

assert abs(clean_fit["vp_vs"] - 1.73) < 0.05, "sanity check on sample Vp/Vs failed"
assert clean_fit["r"] > 0.99, "sanity check on sample correlation failed"
print("\\nSelf-check passed — the Wadati foundation is runnable.")
"""

INGEST_IMPORT_CELL = """\
# --- Ingestion modules (pure Python, sit next to this notebook) ---
import sys
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

from wadati_ingest import (
    CANONICAL_COLUMNS,
    MIN_PAIRS_PER_EVENT,
    SCHEMA_DOC,
    format_timestamp,
    load_csv_picks,
    load_csv_text,
    load_picks,
    load_quakeml_picks,
    write_canonical_csv,
)
import wadati_samples as samples

print(SCHEMA_DOC)
print("canonical columns:", ", ".join(CANONICAL_COLUMNS))
print("default minimum usable pairs per event:", MIN_PAIRS_PER_EVENT)
"""

SAMPLE_WRITE_CELL = """\
# --- Write the sample datasets that the loaders will read back ---
DATA_DIR = NB_DIR / "data"
csv_paths = samples.write_sample_csvs(DATA_DIR)
quakeml_path = samples.write_sample_quakeml(DATA_DIR / "sample_catalog.xml")

for label, path in {**csv_paths, "quakeml": quakeml_path}.items():
    print(f"{label:12s} {path.relative_to(NB_DIR)}  ({path.stat().st_size:,} bytes)")

print()
print(samples.sample_long_csv().splitlines()[0])
for line in samples.sample_long_csv().splitlines()[1:5]:
    print(line)
"""

CSV_LOAD_CELL = '''\
def ingest_panel(result, title: str) -> pn.Column:
    """Field-notebook rendering of one ingestion result."""
    tone = THEME["red"] if result.errors else THEME["teal"]
    verdict = (
        f"{len(result.errors)} error(s) · {len(result.warnings)} warning(s)"
        if result.issues else "clean — no validation messages"
    )
    return pn.Column(
        pn.pane.Markdown(f"**{title}**  \\n`{result.source}`", styles={"color": THEME["ink"]}),
        pn.pane.Markdown(
            f"{len(result.usable_pairs)} usable of {len(result.pairs)} normalized pairs · {verdict}",
            styles={"color": tone},
        ),
        pn.pane.DataFrame(pd.DataFrame(result.event_summary()), index=False, width=980),
        styles={"background": THEME["paper"], "padding": "10px",
                "border": f"1px solid {THEME['rule']}"},
    )


long_result = load_csv_picks(csv_paths["long"])
print(long_result.report())

canonical = long_result.to_dataframe()
display(canonical.head(8))
ingest_panel(long_result, "CSV · long layout, absolute timestamps")
'''

SUMMARY_CELL = """\
# --- Event and station summary tables -------------------------------------
events_frame = pd.DataFrame(long_result.event_summary())
stations_frame = pd.DataFrame(long_result.station_summary())

print(f"{len(events_frame)} event(s), {len(stations_frame)} station(s) ingested")
display(events_frame)
display(stations_frame)
"""

TRAVEL_TIME_CELL = """\
# --- Same events supplied as numeric travel times (no absolute clock) ------
travel_result = load_csv_picks(csv_paths["travel_time"])
print(travel_result.report())

keys = ["event_id", "station_id", "p_travel_time", "s_minus_p"]
lhs = long_result.to_dataframe()[keys].sort_values(keys[:2]).reset_index(drop=True)
rhs = travel_result.to_dataframe()[keys].sort_values(keys[:2]).reset_index(drop=True)
max_drift = float((lhs[keys[2:]] - rhs[keys[2:]]).abs().to_numpy().max())
print(f"\\nmax drift between the timestamp and travel-time layouts: {max_drift:.4f} s")
assert max_drift < 0.01, "the two CSV layouts must normalize to the same numbers"
"""

QUAKEML_CELL = """\
# --- QuakeML ingestion (ObsPy; explicit format is required) ----------------
qml_result = load_quakeml_picks(quakeml_path)
print(qml_result.report())

display(qml_result.to_dataframe().head(8))
ingest_panel(qml_result, "QuakeML · picks paired per event + station")
"""

VALIDATION_CELL = """\
# --- Every validation code, raised by the real loader ----------------------
problem_result = load_csv_text(samples.PROBLEM_CSV, source="csv:sample_picks_problems.csv")
print(problem_result.report())

codes = sorted({issue.code for issue in problem_result.issues})
print("\\ncodes raised:", ", ".join(codes))

issue_frame = pd.DataFrame(
    [
        {
            "level": issue.level,
            "code": issue.code,
            "event_id": issue.event_id,
            "station_id": issue.station_id,
            "message": issue.message,
        }
        for issue in problem_result.issues
    ]
)


def _tint(row):
    color = THEME["red"] if row["level"] == "error" else THEME["slate"]
    return [f"color: {color}"] * len(row)


display(issue_frame.style.apply(_tint, axis=1))
"""

UPLOAD_CELL = '''\
# --- Upload or point at a file: both go through the same loader ------------
upload = pn.widgets.FileInput(accept=".csv,.xml,.quakeml", name="Pick table or QuakeML")
path_box = pn.widgets.TextInput(
    name="…or a path on disk",
    value=str(csv_paths["long"].relative_to(NB_DIR)),
    placeholder="data/sample_picks_long.csv",
)


def ingest_view(uploaded, filename, path_text):
    """Upload wins when present, otherwise the path box is used."""
    if uploaded:
        suffix = Path(filename or "upload.csv").suffix.lower()
        scratch = NB_DIR / "data" / f"_uploaded{suffix or '.csv'}"
        scratch.write_bytes(uploaded)
        result = load_picks(scratch)
    elif path_text.strip():
        target = Path(path_text.strip())
        target = target if target.is_absolute() else NB_DIR / target
        if not target.exists():
            return pn.pane.Markdown(
                f"`{target}` does not exist — check the path.",
                styles={"color": THEME["red"]},
            )
        result = load_picks(target)
    else:
        return pn.pane.Markdown(
            "Upload a `.csv` / `.xml` file, or type a path.",
            styles={"color": THEME["slate"]},
        )
    return pn.Column(
        ingest_panel(result, "Ingested"),
        pn.pane.Markdown(
            "```\\n" + result.report() + "\\n```",
            styles={"color": THEME["slate"]},
        ),
    )


pn.Column(
    pn.pane.Markdown("### Ingest a pick table", styles={"color": THEME["ink"]}),
    pn.Row(upload, path_box),
    pn.bind(ingest_view, upload.param.value, upload.param.filename, path_box.param.value),
    styles={"background": THEME["paper_deep"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
'''

INGESTED_PLOT_CELL = '''\
def wadati_frame(result, event_id: str) -> pd.DataFrame:
    """Canonical pairs -> the ts_p / s_minus_p frame the plot helpers expect."""
    frame = pd.DataFrame([p.as_row() for p in result.event_pairs(event_id)])
    frame = frame.rename(columns={"station_id": "station", "p_travel_time": "ts_p"})
    return frame[["station", "ts_p", "s_minus_p"]].reset_index(drop=True)


# Largest usable event from the ingested QuakeML catalog.
best_event = max(
    {p.event_id for p in qml_result.usable_pairs},
    key=lambda eid: len(qml_result.event_pairs(eid)),
)
ingested = wadati_frame(qml_result, best_event)

raw_ingested_fit = wadati_fit(ingested)
ingested["rejected"] = flag_outliers(ingested, raw_ingested_fit)
ingested_fit = wadati_fit(ingested.loc[~ingested["rejected"]])

pn.Column(
    pn.pane.Markdown(
        f"### Event `{best_event}` — ingested from QuakeML\\n"
        f"Retained **{int((~ingested['rejected']).sum())}** of **{len(ingested)}** station pairs.",
        styles={"color": THEME["ink"]},
    ),
    pn.pane.Matplotlib(
        wadati_plot(ingested, ingested_fit, title=f"Wadati diagram · {best_event}"),
        tight=True, format="svg",
    ),
    pn.pane.Markdown(
        f"`Vp/Vs = {ingested_fit['vp_vs']:.3f} ± {ingested_fit['stderr']:.3f}`  ·  "
        f"`r = {ingested_fit['r']:.4f}`  ·  rejected: "
        f"{', '.join(ingested.loc[ingested['rejected'], 'station']) or 'none'}",
        styles={"color": THEME["slate"]},
    ),
    styles={"background": THEME["paper"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
'''

DISCOVERY_IMPORT_CELL = """\
# --- Real FDSN discovery helpers (no network access at import time) --------
from wadati_discovery import (
    BBOX_DOC,
    DEFAULT_MIN_MAGNITUDE,
    DEFAULT_WINDOW_DAYS,
    EVENT_COLUMNS,
    EVENT_PROVIDER,
    HISPANIOLA_BBOX,
    LO_NETWORK,
    LO_NETWORK_DESCRIPTION,
    STATION_COLUMNS,
    STATION_PROVIDER,
    default_window,
    fetch_hispaniola_events,
    fetch_lo_stations,
    validate_bbox,
    validate_window,
)

print(BBOX_DOC)
print(LO_NETWORK_DESCRIPTION)
print("station table columns:", ", ".join(STATION_COLUMNS))
print("event table columns:  ", ", ".join(EVENT_COLUMNS))
"""

DISCOVERY_UI_CELL = '''\
def discovery_note(text: str, tone: str = "slate") -> pn.pane.Markdown:
    return pn.pane.Markdown(text, styles={"color": THEME[tone]})


def discovery_panel(result, title: str, table_width: int = 980) -> pn.Column:
    """Field-notebook rendering of one discovery result (teal ok / red error)."""
    tone = "red" if result.errors else "teal"
    body = [
        discovery_note(f"**{title}** · provider `{result.provider}`", "ink"),
        discovery_note(
            f"{len(result.rows)} row(s) · {len(result.errors)} error(s) · "
            f"{len(result.warnings)} warning(s)",
            tone,
        ),
    ]
    if result.rows:
        body.append(
            pn.pane.DataFrame(result.to_dataframe(), index=False, width=table_width)
        )
    body.append(discovery_note("```\\n" + result.report() + "\\n```"))
    return pn.Column(
        *body,
        styles={"background": THEME["paper"], "padding": "10px",
                "border": f"1px solid {THEME['rule']}"},
    )


# --- Controls -------------------------------------------------------------
# Nothing below queries a provider until the corresponding button is pressed.
win_start, win_end = default_window(DEFAULT_WINDOW_DAYS)

station_window = pn.widgets.DatetimeRangeInput(
    name="Station metadata active in UTC window",
    value=(win_start.datetime, win_end.datetime),
)
station_button = pn.widgets.Button(
    name=f"Query {LO_NETWORK} stations from {STATION_PROVIDER}", button_type="primary"
)
station_picker = pn.widgets.MultiChoice(
    name="Selected LO / OSPL stations (Dominican Republic)", options=[], value=[],
)
station_out = pn.Column(
    discovery_note("Press the button to query EARTHSCOPE for network LO (OSPL).")
)

STATE = {"stations": None, "events": None}


def run_station_query(_event=None):
    station_button.loading = True
    try:
        start, end = station_window.value
        result = fetch_lo_stations(starttime=start, endtime=end, bbox=HISPANIOLA_BBOX)
        STATE["stations"] = result
        options = result.options()
        station_picker.options = options
        station_picker.value = list(options.values())[:6]
        station_out.objects = [discovery_panel(result, "LO stations · OSPL, Dominican Republic")]
    finally:
        station_button.loading = False


station_button.on_click(run_station_query)

pn.Column(
    pn.pane.Markdown(
        "### Stations — network LO (OSPL, Dominican Republic)",
        styles={"color": THEME["ink"]},
    ),
    discovery_note(
        "Hispaniola is shared by the **Dominican Republic** and **Haiti**. "
        "Network **LO** is the *Observatorio Sismológico Politécnico Loyola* "
        "(**OSPL**) of the Dominican Republic and is served by **EARTHSCOPE** only."
    ),
    pn.Row(station_window, station_button),
    station_picker,
    station_out,
    styles={"background": THEME["paper_deep"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
'''

DISCOVERY_EVENTS_CELL = """\
event_window = pn.widgets.DatetimeRangeInput(
    name="Event origin time, UTC window",
    value=(win_start.datetime, win_end.datetime),
)
mag_input = pn.widgets.FloatSlider(
    name="Minimum magnitude", start=1.0, end=7.0, step=0.1,
    value=DEFAULT_MIN_MAGNITUDE,
)
lat_slider = pn.widgets.EditableRangeSlider(
    name="Latitude (N)", start=14.0, end=23.0, step=0.1,
    value=(HISPANIOLA_BBOX[0], HISPANIOLA_BBOX[1]),
)
lon_slider = pn.widgets.EditableRangeSlider(
    name="Longitude (E, west is negative)", start=-80.0, end=-63.0, step=0.1,
    value=(HISPANIOLA_BBOX[2], HISPANIOLA_BBOX[3]),
)
event_button = pn.widgets.Button(
    name=f"Query regional events from {EVENT_PROVIDER}", button_type="primary"
)
event_picker = pn.widgets.MultiSelect(
    name="Selected earthquakes (Hispaniola and surroundings)",
    options=[], value=[], size=10,
)
event_out = pn.Column(
    discovery_note("Press the button to query USGS for the Hispaniola region.")
)


def run_event_query(_event=None):
    event_button.loading = True
    try:
        start, end = event_window.value
        bbox = (lat_slider.value[0], lat_slider.value[1],
                lon_slider.value[0], lon_slider.value[1])
        result = fetch_hispaniola_events(
            starttime=start, endtime=end,
            minmagnitude=mag_input.value, bbox=bbox,
        )
        STATE["events"] = result
        options = result.options()
        event_picker.options = options
        event_picker.value = list(options.values())[:3]
        event_out.objects = [
            discovery_panel(result, "Regional earthquakes · Hispaniola (DR + Haiti) and offshore")
        ]
    finally:
        event_button.loading = False


event_button.on_click(run_event_query)

pn.Column(
    pn.pane.Markdown(
        "### Earthquakes — Hispaniola and its surrounding region",
        styles={"color": THEME["ink"]},
    ),
    discovery_note(
        "Default box **16.5–20.5 N, 75.5–67.0 W** — the whole island (Dominican "
        "Republic and Haiti) plus nearby offshore seismicity."
    ),
    pn.Row(event_window, mag_input),
    pn.Row(lat_slider, lon_slider),
    event_button,
    event_picker,
    event_out,
    styles={"background": THEME["paper_deep"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
"""

DISCOVERY_SELECTION_CELL = """\
# --- What the user actually selected (empty until a query is run) ----------
def selection_frames():
    stations = STATE["stations"]
    events = STATE["events"]
    station_rows = stations.select(station_picker.value) if stations else []
    event_rows = events.select(event_picker.value) if events else []
    return (
        pd.DataFrame(station_rows, columns=STATION_COLUMNS),
        pd.DataFrame(event_rows, columns=EVENT_COLUMNS),
    )


selected_stations, selected_events = selection_frames()
print(f"{len(selected_stations)} station(s) and {len(selected_events)} event(s) selected")
if selected_stations.empty and selected_events.empty:
    print("Nothing selected yet — run the two queries above, then re-run this cell.")
display(selected_stations)
display(selected_events)
"""

DISCOVERY_SELFCHECK_CELL = """\
# --- Discovery self-check (offline-safe: no provider is contacted) ---------
assert HISPANIOLA_BBOX == (16.5, 20.5, -75.5, -67.0), "default region changed"
assert validate_bbox(HISPANIOLA_BBOX) == HISPANIOLA_BBOX

# Malformed ranges are rejected with an actionable message, not a traceback.
for bad in [("2024-06-01", "2024-05-01"), ("not-a-date", "2024-05-01")]:
    try:
        validate_window(*bad)
    except ValueError as exc:
        print(f"rejected {bad}: {exc}")
    else:
        raise AssertionError(f"{bad} should not validate")

# A bad magnitude range never leaves the process: the result carries the error.
bad_result = fetch_hispaniola_events(
    starttime="2024-06-01", endtime="2024-05-01", minmagnitude=3.0
)
assert bad_result.errors and bad_result.errors[0].code == "malformed_range"
assert not bad_result.rows
print("malformed event window ->", bad_result.errors[0].message)

# An unreachable provider is reported as offline / unreachable, not raised.
offline = fetch_lo_stations(provider="http://localhost:9", bbox=HISPANIOLA_BBOX)
assert offline.errors, "an unreachable provider must produce an error message"
print("unreachable provider ->", offline.errors[0].code)

print("\\nDiscovery self-check passed — every failure path is actionable.")
"""

INGEST_SELFCHECK_CELL = """\
# --- Ingestion self-check (asserts the documented behaviour) ---------------
assert list(canonical.columns) == CANONICAL_COLUMNS, "canonical column order changed"
assert not long_result.errors, "the clean sample CSV must ingest without errors"
assert len(long_result.event_ids) == 4, "expected four sample events"
assert (canonical["s_minus_p"] > 0).all(), "S-P must be strictly positive"
assert (canonical["p_travel_time"] > 0).all(), "P travel time must be positive"

# QuakeML: the duplicate pick and the origin without a time are both caught.
qml_codes = {issue.code for issue in qml_result.issues}
assert "duplicate_phase" in qml_codes, "duplicate P pick was not detected"
assert "missing_origin_time" in qml_codes, "origin without a time was not detected"
assert "gr2024noorigin" not in {p.event_id for p in qml_result.pairs}

# The problem CSV exercises every remaining code.
problem_codes = {issue.code for issue in problem_result.issues}
for code in (
    "duplicate_phase", "missing_p", "missing_s", "s_not_after_p",
    "malformed_time", "unknown_phase", "missing_origin_time", "insufficient_pairs",
):
    assert code in problem_codes, f"validation code {code} was not raised"

# Ingested QuakeML recovers the true Vp/Vs of the synthetic medium.
assert abs(ingested_fit["vp_vs"] - samples.TRUE_VP_VS) < 0.06, "ingested Vp/Vs drifted"
assert ingested_fit["r"] > 0.99, "ingested correlation too low"

out = write_canonical_csv(long_result, NB_DIR / "data" / "canonical_picks.csv")
print(f"canonical table written to {out.relative_to(NB_DIR)}")
print("Ingestion self-check passed — step 2 is complete.")
"""

SUBSET_IMPORT_CELL = """\
# --- Exhaustive subset search (pure Python + scipy, no notebook state) -----
from wadati_subset import (
    ABSOLUTE_MIN_STATIONS,
    CONFIG_DOC,
    DEFAULT_MIN_CORRELATION,
    EVENT_QC_COLUMNS,
    PICK_QC_COLUMNS,
    REASON_TEXT,
    DatasetQC,
    SearchConfig,
    WadatiPoint,
    event_qc_csv_text,
    fit_subset,
    pick_qc_csv_text,
    points_from_pairs,
    run_subset_search,
    search_event,
    write_event_qc_csv,
    write_pick_qc_csv,
)
import wadati_subset_examples as subset_examples

print(CONFIG_DOC)
print("absolute minimum stations:", ABSOLUTE_MIN_STATIONS)
print("default correlation threshold:", DEFAULT_MIN_CORRELATION)
"""

SUBSET_CONFIG_CELL = """\
# --- Configuration examples ------------------------------------------------
CONFIGS = {
    "published defaults": SearchConfig(),
    "stricter correlation": SearchConfig(min_correlation=0.95),
    "six stations minimum": SearchConfig(min_stations=6),
    "correlation only (no Vp/Vs bounds)": SearchConfig(vp_vs_min=None, vp_vs_max=None),
    "narrow crustal window": SearchConfig(vp_vs_min=1.60, vp_vs_max=1.85),
    "remove at most two picks": SearchConfig(max_removals=2),
    "below the floor (clamped to 3)": SearchConfig(min_stations=2),
}

config_rows = []
for label, cfg in CONFIGS.items():
    normalized, notes = cfg.normalized()
    config_rows.append({
        "configuration": label,
        "min |r|": normalized.min_correlation,
        "min stations": normalized.min_stations,
        "Vp/Vs min": normalized.vp_vs_min,
        "Vp/Vs max": normalized.vp_vs_max,
        "max removals": normalized.max_removals,
        "note": "; ".join(notes) or "—",
    })

SEARCH_CONFIG = CONFIGS["published defaults"]
print("active configuration:", SEARCH_CONFIG.describe())
display(pd.DataFrame(config_rows))
"""

SUBSET_EVENT_CELL = """\
# --- One event, start to finish -------------------------------------------
qc_event_id = best_event
qc_points = points_from_pairs(qml_result.event_pairs(qc_event_id))
qc = search_event(qc_event_id, qc_points, SEARCH_CONFIG, source=qml_result.source)

print(qc.report())

qc_frame = qc.plot_frame()
qc_fit = qc.plot_fit()
tone = THEME["teal"] if qc.accepted else THEME["red"]

pn.Column(
    pn.pane.Markdown(
        f"### Event `{qc.event_id}` — exhaustive subset search\\n"
        f"**{qc.status.upper()}** · `{qc.reason}` · retained "
        f"**{qc.retained_stations}** of **{qc.original_stations}** stations at "
        f"search depth **{qc.search_depth}**.",
        styles={"color": tone},
    ),
    pn.pane.Matplotlib(
        wadati_plot(qc_frame, qc_fit, title=f"Wadati subset search · {qc.event_id}"),
        tight=True, format="svg",
    ),
    pn.pane.Markdown(
        f"`Vp/Vs = {qc.fit.vp_vs:.3f} ± {qc.fit.vp_vs_stderr:.3f}`  ·  "
        f"`r = {qc.fit.pearson_r:.4f}`  ·  `r² = {qc.fit.r_squared:.4f}`  ·  "
        f"`rmse = {qc.fit.rmse:.3f}`  ·  outliers: "
        f"{', '.join(qc.outlier_stations) or 'none'}  ·  "
        f"{qc.combinations_evaluated:,} combination(s) evaluated",
        styles={"color": THEME["slate"]},
    ),
    styles={"background": THEME["paper"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
"""

SUBSET_DATASET_CELL = """\
# --- Dataset-wide run over the canonical IngestResult ---------------------
dataset_qc = run_subset_search(qml_result, SEARCH_CONFIG)
csv_dataset_qc = run_subset_search(long_result, SEARCH_CONFIG)

print(dataset_qc.report())
print()
print(csv_dataset_qc.report())


def _qc_tint(row):
    color = THEME["teal"] if row["status"] == "accepted" else THEME["red"]
    return [f"color: {color}"] * len(row)


event_qc_frame = dataset_qc.event_dataframe()
pick_qc_frame = dataset_qc.pick_dataframe()

display(
    event_qc_frame[[
        "event_id", "status", "reason", "original_stations", "retained_stations",
        "removed_stations", "search_depth", "combinations_evaluated",
        "pearson_r", "r_squared", "rmse", "vp_vs", "vp_vs_stderr",
    ]].style.apply(_qc_tint, axis=1)
)
display(pd.DataFrame([dataset_qc.summary()]))
display(pick_qc_frame.head(12))
"""

SUBSET_EXPORT_CELL = """\
# --- Downloadable quality-control results ---------------------------------
import io

qc_dir = NB_DIR / "data"
event_csv = write_event_qc_csv(dataset_qc, qc_dir / "wadati_qc_events.csv")
pick_csv = write_pick_qc_csv(dataset_qc, qc_dir / "wadati_qc_picks.csv")
print(f"per-event QC   {event_csv.relative_to(NB_DIR)}  "
      f"({event_csv.stat().st_size:,} bytes, {len(EVENT_QC_COLUMNS)} columns)")
print(f"per-pick QC    {pick_csv.relative_to(NB_DIR)}  "
      f"({pick_csv.stat().st_size:,} bytes, {len(PICK_QC_COLUMNS)} columns)")

event_download = pn.widgets.FileDownload(
    callback=lambda: io.StringIO(event_qc_csv_text(dataset_qc)),
    filename="wadati_qc_events.csv",
    label="Download per-event QC (CSV)",
    button_type="primary",
)
pick_download = pn.widgets.FileDownload(
    callback=lambda: io.StringIO(pick_qc_csv_text(dataset_qc)),
    filename="wadati_qc_picks.csv",
    label="Download per-pick retained / outlier table (CSV)",
    button_type="primary",
)

pn.Column(
    pn.pane.Markdown("### Export quality-control results", styles={"color": THEME["ink"]}),
    pn.pane.Markdown(
        f"{len(dataset_qc.results)} event(s) · "
        f"{len(dataset_qc.accepted)} accepted / {len(dataset_qc.rejected)} rejected · "
        f"{len(pick_qc_frame)} pick row(s)",
        styles={"color": THEME["slate"]},
    ),
    pn.Row(event_download, pick_download),
    styles={"background": THEME["paper_deep"], "padding": "12px",
            "border": f"1px solid {THEME['rule']}"},
)
"""

SUBSET_EXAMPLES_CELL = """\
# --- Worked examples: all-pass, one/two outliers, anomalous Vp/Vs, hopeless
example_rows = subset_examples.check_examples()
display(pd.DataFrame(example_rows))

for example, result in subset_examples.run_examples():
    print("=" * 72)
    print(f"{example.key} — {example.title}")
    print(example.description)
    print("-" * 72)
    print(result.report())
    print()

example_panels = []
for example, result in subset_examples.run_examples():
    tone = THEME["teal"] if result.accepted else THEME["red"]
    example_panels.append(
        pn.Column(
            pn.pane.Markdown(
                f"**{example.title}** — `{result.status}` / `{result.reason}`",
                styles={"color": tone},
            ),
            pn.pane.Matplotlib(
                wadati_plot(result.plot_frame(), result.plot_fit(), title=example.key),
                tight=True, format="svg",
            ),
            styles={"background": THEME["paper"], "padding": "10px",
                    "border": f"1px solid {THEME['rule']}"},
        )
    )

pn.GridBox(*example_panels, ncols=2)
"""

SUBSET_SELFCHECK_CELL = """\
# --- Subset-search self-check (fully offline) ------------------------------
assert DEFAULT_MIN_CORRELATION == 0.9, \"the documented default threshold changed\"
assert ABSOLUTE_MIN_STATIONS == 3, \"the absolute station floor changed\"
assert SearchConfig(min_stations=1).normalized()[0].min_stations == 3

# Larger subsets are always preferred: an all-pass event must remove nothing.
all_pass = next(e for e in subset_examples.examples() if e.key == \"all_pass\").run()
assert all_pass.search_depth == 0 and not all_pass.removed_stations

# Correlation failure and anomalous Vp/Vs are distinct, reported reasons.
reasons = {e.key: e.run().reason for e in subset_examples.examples()}
assert reasons[\"anomalous_vp_vs\"] == \"rejected_vp_vs\"
assert reasons[\"irrecoverable\"] == \"rejected_correlation\"
assert reasons[\"one_outlier\"] == reasons[\"two_outliers\"] == \"accepted_after_removal\"

# Depth ordering: two outliers cannot be repaired at depth 1.
two = next(e for e in subset_examples.examples() if e.key == \"two_outliers\").run()
assert two.search_depth == 2, two.search_depth

# Every documented column is present in both exports.
assert list(event_qc_frame.columns) == EVENT_QC_COLUMNS
assert list(pick_qc_frame.columns) == PICK_QC_COLUMNS
assert set(pick_qc_frame[\"state\"]) <= {\"retained\", \"outlier\"}
assert all(reason in REASON_TEXT for reason in event_qc_frame[\"reason\"])

subset_examples.check_examples()
print(\"Subset-search self-check passed — step 3 is complete.\")
"""


DASHBOARD_IMPORT_CELL = """\
# --- Dashboard module (pure logic + Panel layer; nothing loaded at import) --
from wadati_dashboard import (
    DASHBOARD_DOC,
    FILTERS,
    SUPPORTED_MODELS,
    THEME as DASH_THEME,
    CandidatePick,
    DashboardSession,
    PickEdit,
    PickEditError,
    SeisBenchError,
    available_weights,
    build_dashboard,
    read_mseed,
    repick_stream,
    self_check as dashboard_self_check,
    stage_candidate_pair,
    wadati_figure,
)

print(DASHBOARD_DOC)
print("status filters:", ", ".join(FILTERS))
print("supported SeisBench model classes:", ", ".join(SUPPORTED_MODELS))
print("amber (revised picks):", DASH_THEME["amber"])
print("no network access and no model weights are touched by this import.")
"""

DASHBOARD_SESSION_CELL = """\
# --- One session: first pass + editable working picks + second pass --------
session = DashboardSession(qml_result, SEARCH_CONFIG)
print(session.report())

first_event = session.event_ids()[0]
print()
print(f"accepted: {session.filtered_event_ids('accepted')}")
print(f"rejected: {session.filtered_event_ids('rejected')}")
display(pd.DataFrame(session.pick_table(first_event)))
"""

DASHBOARD_CELL = """\
# --- The embedded dashboard ------------------------------------------------
# Left: the real Wadati plot, event navigation + status filter, the editable
#       pick table, the second pass and the downloadable QC exports.
# Right: optional waveform-assisted SeisBench re-picking and the real
#       LO / OSPL + Hispaniola FDSN selection. Every query and every model
#       load happens only when its button is pressed.
dashboard = build_dashboard(session, export_dir=NB_DIR / "data")
dashboard
"""

DASHBOARD_EDIT_CELL = """\
# --- The same editing API, driven from code (identical validation) ---------
station = session.station_ids(first_event)[0]
pick = session.pick(first_event, station)
print(f"{station}: t_P = {pick.p_travel_time:.3f} s   S-P = {pick.s_minus_p:.3f} s")

# Every invalid edit is refused with an actionable message.
for kwargs in [
    {"p_travel_time": -2.0},
    {"s_minus_p": 0.0},
    {"s_time": "not-a-timestamp"},
    {},
]:
    try:
        session.stage_edit(first_event, station, **kwargs)
    except PickEditError as exc:
        print(f"rejected {kwargs}: {exc}")

# A valid edit is staged (nothing changes yet), then applied, then reverted.
edit = session.stage_edit(
    first_event, station, s_minus_p=pick.s_minus_p + 1.4,
    note="S re-read on the transverse component",
)
print("\\nstaged  :", edit.describe())
print("revised?", session.pick(first_event, station).revised, "(still the original)")

session.apply_pending(first_event)
print("applied :", session.pick(first_event, station).revised,
      "provenance:", session.pick(first_event, station).provenance)

revised_event = session.rerun_event(first_event)
print(f"\\nsecond pass: {revised_event.status} / {revised_event.reason}  "
      f"Vp/Vs = {revised_event.fit.vp_vs:.3f}")
display(pd.DataFrame(session.comparison_rows(first_event)))

session.undo_last(first_event)
session.reset_event(first_event)
print("undone and reset — the ingested picks are back:",
      not session.pick(first_event, station).revised)
"""

DASHBOARD_SECOND_PASS_CELL = """\
# --- Second pass over the whole dataset, side by side ---------------------
revised_dataset = session.rerun_all()
print(revised_dataset.report())

display(pd.DataFrame(session.dataset_comparison()))

exports = session.write_exports(NB_DIR / \"data\")
for label, path in exports.items():
    print(f"{label:18s} {path.relative_to(NB_DIR)}  ({path.stat().st_size:,} bytes)")
"""

DASHBOARD_SEISBENCH_CELL = """\
# --- Optional SeisBench assist, driven from code (needs network the first
#     time so the weights can be cached in ~/.seisbench/models) -------------
MSEED_PATH = NB_DIR / \"data\" / \"waveforms.mseed\"   # put a real file here
MODEL_NAME, WEIGHTS = \"PhaseNet\", \"stead\"

try:
    print(f"{MODEL_NAME} weights:", \", \".join(available_weights(MODEL_NAME)))
except SeisBenchError as exc:
    print(f"weight listing unavailable — {exc.code}: {exc.message}")

if MSEED_PATH.exists():
    try:
        stream = read_mseed(MSEED_PATH)
        candidates = repick_stream(stream, MODEL_NAME, WEIGHTS, min_confidence=0.3)
        display(pd.DataFrame([c.as_row() for c in candidates]))
        print(f"{len(candidates)} candidate pick(s) — select one explicitly with "
              \"stage_candidate_pair(...) before it touches any pick\")
    except SeisBenchError as exc:
        print(f"{exc.code}: {exc.message}")
else:
    print(f"no waveform at {MSEED_PATH} — upload one in the dashboard above, or "
          \"drop a MiniSEED file at that path. There are no synthetic \"
          \"predictions: without a real waveform nothing is picked.\")
"""

DASHBOARD_SELFCHECK_CELL = """\
# --- Dashboard self-check (fully offline; loads no model weights) ----------
summary = dashboard_self_check()
print(summary)

assert set(FILTERS) == {\"all\", \"accepted\", \"rejected\", \"edited\", \"revised\"}
assert SUPPORTED_MODELS == (\"PhaseNet\", \"EQTransformer\", \"GPD\")
assert DASH_THEME[\"amber\"] == \"#B57415\", \"the revised-pick accent changed\"

# Nothing is ever applied without an explicit selection.
try:
    stage_candidate_pair(session, first_event, station, None, None)
except PickEditError as exc:
    print(\"no selection ->\", exc)
else:
    raise AssertionError(\"model picks must never be applied implicitly\")

# Truthful errors, never a mock prediction.
for bad in [\"NotAModel\", \"MagicPicker\"]:
    try:
        available_weights(bad)
    except SeisBenchError as exc:
        print(f\"{bad} -> {exc.code}\")

print(f\"staged edits pending: {len(session.pending)} · \"
      f\"applied edits: {len(session.applied)} · \"
      f\"reruns: {len(session.revised_events)}\")
print(\"\\nDashboard self-check passed
"""


def _notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Interactive Wadati Quality-Control Notebook\n"
            "**Foundation notebook — step 1 of the workflow.**\n\n"
            "A Wadati diagram plots the S–P interval against the P travel time for "
            "every station that recorded one event. In a homogeneous medium the "
            "points fall on a straight line whose slope is `Vp/Vs − 1`, so a poor "
            "correlation is direct evidence of a mis-identified phase pick.\n\n"
            "This notebook establishes the runnable foundation: environment check, "
            "scientific imports, the shared field-notebook theme, a sample "
            "phase-pick table, the regression helpers, and the Wadati centerpiece "
            "plot rendered through Panel.\n\n"
            "---\n"
            "### Workflow roadmap\n"
            "1. **Foundation (this notebook)** — environment, theme, sample event, Wadati plot.\n"
            "2. **Ingestion (this notebook)** — validated CSV pick tables and ObsPy QuakeML catalogs, "
            "plus real FDSN discovery of LO / OSPL stations (Dominican Republic) and "
            "Hispaniola-region earthquakes.\n"
            "3. **Subset search (this notebook)** — exhaustive removal search with min "
            "correlation (default 0.9), min stations (floor 3) and Vp/Vs bounds, plus "
            "downloadable per-event and per-pick QC results.\n"
            "4. **Panel dashboard (this notebook)** — event-by-event navigation and "
            "status filtering, the linked Wadati plot and editable pick table, manual "
            "re-picking with validation / provenance / undo / reset, optional "
            "waveform-assisted SeisBench re-picking on the CPU, a second pass, and "
            "side-by-side original-versus-revised QC exports.\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 1 · Initialise Panel for embedded notebook use\n"
            "`pn.extension()` must run **before** any Panel object is displayed. "
            "`pyviz_comms` lives in the same environment as JupyterLab, so no lab "
            "extension install is required."
        ),
        nbf.v4.new_code_cell(
            "import panel as pn\n\n"
            'pn.extension(sizing_mode="stretch_width")\n'
            'print(f"Panel {pn.__version__} extension loaded for notebook output.")'
        ),
        nbf.v4.new_markdown_cell(
            "## 2 · Scientific stack\n"
            "ObsPy is imported here so QuakeML ingestion in step 2 needs no new "
            "imports. SeisBench/torch are checked lazily at the end of the notebook."
        ),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib as mpl\n"
            "from matplotlib.figure import Figure\n"
            "from scipy.stats import linregress\n"
            "from obspy import read_events  # noqa: F401 - used by the ingestion step\n\n"
            'pd.set_option("display.precision", 3)\n'
            'print("Scientific stack imported.")'
        ),
        nbf.v4.new_markdown_cell(
            "## 3 · Field-notebook theme\n"
            "One palette drives Matplotlib and Panel: warm mineral paper, charcoal "
            "and slate type, thin geological rule lines, deep **teal** for retained "
            "picks and restrained **seismic red** for rejected picks."
        ),
        nbf.v4.new_code_cell(THEME_CELL),
        nbf.v4.new_markdown_cell(
            "## 4 · Sample event\n"
            "A synthetic but scientifically shaped single-event pick table with two "
            "deliberate mis-picks, so the retained/rejected states are visible."
        ),
        nbf.v4.new_code_cell(SAMPLE_CELL),
        nbf.v4.new_markdown_cell(
            "## 5 · Regression and outlier flagging\n"
            "`wadati_fit` returns slope, intercept, Pearson *r* and the implied "
            "`Vp/Vs`. `flag_outliers` marks picks whose residual exceeds "
            "*n·σ* of the residual spread — the seed of the exhaustive subset "
            "search in step 3."
        ),
        nbf.v4.new_code_cell(FIT_CELL),
        nbf.v4.new_markdown_cell(
            "## 6 · Centerpiece — the Wadati diagram\n"
            "The plot is the primary instrument of this workflow: teal for retained "
            "pairs, seismic-red crosses for rejected ones, and the fitted line "
            "annotated with `Vp/Vs` and *r*."
        ),
        nbf.v4.new_code_cell(PLOT_CELL),
        nbf.v4.new_markdown_cell(
            "## 7 · Environment self-check\n"
            "Confirms the pinned versions and asserts the sample fit recovers the "
            "true `Vp/Vs`. Everything above runs top-to-bottom with no network access."
        ),
        nbf.v4.new_code_cell(VERIFY_CELL),
        nbf.v4.new_markdown_cell(
            "---\n# Step 2 · Validated ingestion and normalization\n"
            "Two input families are supported and both normalize onto **one canonical "
            "schema** — one row per event/station P–S pair:\n\n"
            "| column | required | meaning |\n| --- | --- | --- |\n"
            "| `event_id` | yes | catalog / origin identifier |\n"
            "| `station_id` | yes | station code, e.g. `BRG` |\n"
            "| `network`, `channel` | no | SEED codes when available |\n"
            "| `source` | yes | provenance, e.g. `csv:picks.csv` |\n"
            "| `origin_time`, `p_time`, `s_time` | no | ISO-8601 UTC, when absolute |\n"
            "| `p_travel_time` | yes | origin-relative P travel time (s) |\n"
            "| `s_minus_p` | yes | S−P interval (s), strictly positive |\n\n"
            "Absolute timestamps **or** numeric travel times are accepted wherever "
            "scientifically valid: with timestamps the S−P interval is differenced "
            "directly and the P travel time is measured against the origin time; with "
            "numeric travel times no clock is needed at all.\n\n"
            "The loaders live in `wadati_ingest.py` next to this notebook — plain "
            "Python, no notebook state, so they are reusable and testable."
        ),
        nbf.v4.new_markdown_cell(
            "## 8 · Canonical schema and loader import\n"
            "`SCHEMA_DOC` is the single source of truth for the schema, the accepted "
            "CSV layouts, the QuakeML contract and every validation code."
        ),
        nbf.v4.new_code_cell(INGEST_IMPORT_CELL),
        nbf.v4.new_markdown_cell(
            "## 9 · Sample data\n"
            "Four Vogtland-style swarm events across up to twelve stations, written "
            "three ways — long CSV with absolute timestamps, wide CSV with numeric "
            "travel times, and a deliberately broken table — plus a QuakeML catalog "
            "generated programmatically with ObsPy. The loaders read these files back "
            "from disk, so the samples exercise the real code path."
        ),
        nbf.v4.new_code_cell(SAMPLE_WRITE_CELL),
        nbf.v4.new_markdown_cell(
            "## 10 · CSV ingestion\n"
            "P and S picks are paired by **event and station**; the report lists every "
            "actionable message raised while reading."
        ),
        nbf.v4.new_code_cell(CSV_LOAD_CELL),
        nbf.v4.new_markdown_cell(
            "### Event and station summaries\n"
            "`event_summary()` reports pair counts, the travel-time span, a median "
            "apparent `Vp/Vs` and whether the event clears the usable-pair threshold. "
            "`station_summary()` aggregates the same pairs by station."
        ),
        nbf.v4.new_code_cell(SUMMARY_CELL),
        nbf.v4.new_markdown_cell(
            "### The same events as numeric travel times\n"
            "A travel-time table carries no clock, so nothing is derived from an origin "
            "time — the normalized numbers must nevertheless agree with the "
            "timestamp layout."
        ),
        nbf.v4.new_code_cell(TRAVEL_TIME_CELL),
        nbf.v4.new_markdown_cell(
            "## 11 · QuakeML ingestion\n"
            "Picks are paired per event and station through `pick.waveform_id`; the "
            "phase comes from `pick.phase_hint` and falls back to the matching origin "
            "arrival phase. P travel times are measured against the **preferred "
            "origin**, falling back to the first available origin that carries a time. "
            "An origin with no time is reported rather than silently dropped."
        ),
        nbf.v4.new_code_cell(QUAKEML_CELL),
        nbf.v4.new_markdown_cell(
            "## 12 · Validation — actionable messages\n"
            "Duplicate phases, a missing P or S, an S that is not later than P, "
            "malformed timestamps, an absent origin time and events below the usable "
            "pair count are all detected and explained. Errors are seismic red, "
            "warnings slate."
        ),
        nbf.v4.new_code_cell(VALIDATION_CELL),
        nbf.v4.new_markdown_cell(
            "## 13 · Upload or path\n"
            "`load_picks` dispatches on the suffix, so an uploaded file and a path on "
            "disk take exactly the same route into the canonical schema."
        ),
        nbf.v4.new_code_cell(UPLOAD_CELL),
        nbf.v4.new_markdown_cell(
            "## 14 · The centerpiece, now fed by real ingestion\n"
            "The Wadati diagram from step 1, rendered from the normalized QuakeML "
            "pairs instead of the in-notebook sample table."
        ),
        nbf.v4.new_code_cell(INGESTED_PLOT_CELL),
        nbf.v4.new_markdown_cell(
            "## 15 · Ingestion self-check\n"
            "Asserts the canonical column order, the clean-input contract, every "
            "validation code, and that the ingested catalog recovers the true `Vp/Vs`."
        ),
        nbf.v4.new_code_cell(INGEST_SELFCHECK_CELL),
        nbf.v4.new_markdown_cell(
            "---\n## 16 · Real data discovery — LO / OSPL stations and Hispaniola events\n"
            "**Hispaniola** is the island shared by the **Dominican Republic** and "
            "**Haiti**. Network **LO** is the *Observatorio Sismológico Politécnico "
            "Loyola* (**OSPL**) of the **Dominican Republic**; its metadata is served "
            "by **EARTHSCOPE** (the only FDSN node that carries LO). Regional "
            "earthquakes come from **USGS**.\n\n"
            "The default region — **16.5–20.5 N, 75.5–67.0 W** — is deliberately "
            "broader than the island so that offshore seismicity along the "
            "Septentrional and Enriquillo fault zones, the Muertos trough and the "
            "western Puerto Rico trench approaches is included. Both the UTC date "
            "range and the magnitude threshold are controls.\n\n"
            "`wadati_discovery.py` performs **no network access at import time**: "
            "every query runs only when its button is pressed, and no-data, "
            "malformed ranges, provider failures and offline errors all come back as "
            "actionable messages on the returned result rather than as tracebacks."
        ),
        nbf.v4.new_code_cell(DISCOVERY_IMPORT_CELL),
        nbf.v4.new_markdown_cell(
            "### Station controls\n"
            "A `MultiChoice` is populated from the **real inventory** returned by "
            "EARTHSCOPE — the selection is never fabricated. Each row carries a "
            "stable `station_uid` (`LO.LOBH`), coordinates, elevation, the site name "
            "and the available channel codes."
        ),
        nbf.v4.new_code_cell(DISCOVERY_UI_CELL),
        nbf.v4.new_markdown_cell(
            "### Event controls\n"
            "A `MultiSelect` is populated from the **real USGS catalog** for the "
            "configured UTC window, magnitude threshold and bounding box. Each row "
            "carries a stable `event_uid`, origin time, epicentre, depth in km and "
            "the preferred magnitude with its type."
        ),
        nbf.v4.new_code_cell(DISCOVERY_EVENTS_CELL),
        nbf.v4.new_markdown_cell(
            "### The selection, as tidy tables\n"
            "These are the rows the next step (the exhaustive subset search) will "
            "consume. They stay empty until the two queries above have been run."
        ),
        nbf.v4.new_code_cell(DISCOVERY_SELECTION_CELL),
        nbf.v4.new_markdown_cell(
            "### Discovery self-check\n"
            "Runs fully offline: it asserts the documented default box and that a "
            "reversed window, an unparsable date and an unreachable provider each "
            "produce an actionable message instead of an exception."
        ),
        nbf.v4.new_code_cell(DISCOVERY_SELFCHECK_CELL),
        nbf.v4.new_markdown_cell(
            "---\n# Step 3 · Exhaustive Wadati subset search\n"
            "For one event, every station with a valid P **and** S pick contributes "
            "one Wadati point. The algorithm is deliberately explicit:\n\n"
            "1. Fit `s_minus_p` against `p_travel_time` over **all** valid station "
            "pairs with `scipy.stats.linregress`.\n"
            "2. If Pearson `|r|` reaches `min_correlation` (**default 0.9**), the "
            "retained station count reaches `min_stations`, and the implied `Vp/Vs` "
            "lies inside the optional `[vp_vs_min, vp_vs_max]` bounds — **accept "
            "every pick**, nothing is removed.\n"
            "3. Otherwise enumerate subsets by removing **one** point, then **two**, "
            "then three … so larger subsets are always tested before smaller ones. "
            "**Stop at the first removal depth** that yields qualifying candidates.\n"
            "4. Choose among those candidates deterministically: highest `|r|`, then "
            "lowest residual error (RMSE), then the stable station ordering of the "
            "removed stations.\n"
            "5. **Reject** the event when no qualifying subset remains at or above "
            "`min_stations`, which is never allowed below **3** — a line through two "
            "points is trivially perfect and carries no information.\n\n"
            "Two scientifically different failures are reported separately: "
            "`rejected_correlation` (the picks are mutually inconsistent) versus "
            "`rejected_vp_vs` (the picks *are* consistent, but the velocity ratio is "
            "anomalous). `rejected_insufficient_stations` and "
            "`rejected_search_truncated` cover the two remaining cases.\n\n"
            "The module is `wadati_subset.py` next to this notebook — plain Python, "
            "reusable and testable, with `wadati_subset_examples.py` holding the "
            "worked examples."
        ),
        nbf.v4.new_markdown_cell(
            "## 17 · Criteria, result fields and configuration\n"
            "`print(CONFIG_DOC)` is the single source of truth for the configuration "
            "knobs, the status/reason codes and the result API."
        ),
        nbf.v4.new_code_cell(SUBSET_IMPORT_CELL),
        nbf.v4.new_markdown_cell(
            "### Configuration examples\n"
            "Every criterion is configurable; `SearchConfig.normalized()` clamps a "
            "`min_stations` below 3 up to the floor and explains why."
        ),
        nbf.v4.new_code_cell(SUBSET_CONFIG_CELL),
        nbf.v4.new_markdown_cell(
            "## 18 · One event, start to finish\n"
            "The Wadati scatter and its fitted line remain the central instrument: "
            "teal for retained pairs, seismic-red crosses for the removed outliers. "
            "The report lists the accepted/rejected status and reason, the original "
            "and retained counts, the removed stations, slope, intercept, Pearson "
            "`r`, `r²`, residual metrics, standard errors, the `Vp/Vs` estimate, the "
            "active thresholds, the number of combinations evaluated and the search "
            "depth."
        ),
        nbf.v4.new_code_cell(SUBSET_EVENT_CELL),
        nbf.v4.new_markdown_cell(
            "## 19 · Dataset-wide run over canonical `IngestResult` data\n"
            "`run_subset_search(ingest_result, config)` groups the canonical pairs by "
            "event and returns a `DatasetQC` with per-event and per-pick tables, a "
            "summary and a printable report. Both the ingested QuakeML catalog and "
            "the CSV table are run through the identical code path."
        ),
        nbf.v4.new_code_cell(SUBSET_DATASET_CELL),
        nbf.v4.new_markdown_cell(
            "## 20 · Downloadable quality-control results\n"
            "Two real exports: the detailed per-event QC table and the per-pick "
            "retained/outlier table. Both are written to `data/` **and** offered as "
            "Panel `FileDownload` widgets driven by the same functions."
        ),
        nbf.v4.new_code_cell(SUBSET_EXPORT_CELL),
        nbf.v4.new_markdown_cell(
            "## 21 · Worked examples\n"
            "Five deterministic events: one that passes outright, one repaired by "
            "removing a single outlier, one that needs two removals (depth 1 is not "
            "enough), one with an excellent correlation but an anomalous `Vp/Vs`, and "
            "one that no subset can rescue."
        ),
        nbf.v4.new_code_cell(SUBSET_EXAMPLES_CELL),
        nbf.v4.new_markdown_cell(
            "## 22 · Subset-search self-check\n"
            "Asserts the published defaults, the station floor, the depth ordering, "
            "the distinction between a correlation rejection and an anomalous "
            "`Vp/Vs`, and the exported column contracts."
        ),
        nbf.v4.new_code_cell(SUBSET_SELFCHECK_CELL),
        nbf.v4.new_markdown_cell(
            "---\n# Step 4 · The embedded Panel dashboard\n"
            "`wadati_dashboard.py` next to this notebook carries two clearly "
            "separated layers:\n\n"
            "* **Logic** — `DashboardSession` holds the first-pass `DatasetQC`, the "
            "editable working picks, a provenance-carrying edit log, and the second "
            "pass over one event or the whole dataset. Every edit is validated, "
            "staged, applied, undone or reset **explicitly**; nothing is ever "
            "silently overwritten. It is plain Python and fully testable offline.\n"
            "* **Panel** — `build_dashboard(session)` assembles the dense, asymmetric "
            "field-notebook workspace: warm mineral paper, charcoal and slate type, "
            "thin geological rule lines, deep **teal** retained / accepted, **seismic "
            "red** outlier / rejected, and **amber** for revised picks. The real "
            "Wadati plot is the centre of the workspace, not a decoration.\n\n"
            "Optional **SeisBench** assistance is lazy: model classes and pretrained "
            "weights are only listed, downloaded and loaded when a button is pressed, "
            "inference runs on the **CPU** through the official "
            "`model.classify(stream)` API, candidate P/S picks are shown with their "
            "real confidence, and a candidate reaches a pick only after you select it "
            "and stage it. Missing packages, an unknown weight name, an empty cache "
            "and an offline machine are reported truthfully — there are no mock "
            "predictions anywhere."
        ),
        nbf.v4.new_markdown_cell(
            "## 23 · Dashboard API\n"
            "`print(DASHBOARD_DOC)` is the single source of truth for the session "
            "API, the edit validation rules and every SeisBench error code."
        ),
        nbf.v4.new_code_cell(DASHBOARD_IMPORT_CELL),
        nbf.v4.new_markdown_cell(
            "## 24 · One session over the ingested catalog\n"
            "The session runs the exhaustive subset search once as the **first pass** "
            "and keeps it immutable, so the original metrics are always available for "
            "the side-by-side comparison."
        ),
        nbf.v4.new_code_cell(DASHBOARD_SESSION_CELL),
        nbf.v4.new_markdown_cell(
            "## 25 · The dashboard\n"
            "Navigate events with the filter and the ◀ / ▶ buttons; the Wadati plot, "
            "the metrics line, the editable pick table, the comparison tables and the "
            "per-event report are all linked to the selected event. Edit a P or S "
            "time (numeric travel times **or** absolute ISO timestamps), stage it, "
            "apply it, then rerun the event or the whole dataset as a second pass and "
            "download the original and revised event / pick QC CSVs."
        ),
        nbf.v4.new_code_cell(DASHBOARD_CELL),
        nbf.v4.new_markdown_cell(
            "## 26 · The same editing API from code\n"
            "The widgets are a thin skin over these calls, so every rule the "
            "dashboard enforces is reproducible in a cell."
        ),
        nbf.v4.new_code_cell(DASHBOARD_EDIT_CELL),
        nbf.v4.new_markdown_cell(
            "## 27 · Second pass and the exports\n"
            "`rerun_all()` re-runs the exhaustive search over the working picks and "
            "`dataset_comparison()` reports the original and revised counts side by "
            "side. All four QC CSVs are written to `data/` and are the same text the "
            "dashboard's download buttons serve."
        ),
        nbf.v4.new_code_cell(DASHBOARD_SECOND_PASS_CELL),
        nbf.v4.new_markdown_cell(
            "## 28 · Waveform-assisted re-picking, from code\n"
            "Point `MSEED_PATH` at a real MiniSEED file (or upload one in the "
            "dashboard). The first weight load downloads to `~/.seisbench/models`; "
            "with no network access and an empty cache the failure is reported as "
            "`weights_unavailable` / `offline` rather than faked."
        ),
        nbf.v4.new_code_cell(DASHBOARD_SEISBENCH_CELL),
        nbf.v4.new_markdown_cell(
            "## 29 · Dashboard self-check\n"
            "Runs fully offline and loads no model weights: it asserts the edit "
            "validation rules, staging / apply / undo / reset, the second pass, the "
            "comparison and export contracts, and that model predictions are never "
            "applied without an explicit selection."
        ),
        nbf.v4.new_code_cell(DASHBOARD_SELFCHECK_CELL),
        nbf.v4.new_markdown_cell(
            "---\n**The workflow is complete.** Environment and theme, validated CSV "
            "and QuakeML ingestion with real LO / OSPL and Hispaniola FDSN discovery, "
            "the exhaustive Wadati subset search with downloadable QC results, and "
            "this embedded Panel dashboard with manual and optional SeisBench-assisted "
            "re-picking plus a compared second pass."
        ),
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (wadati-qc)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main() -> None:
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        env_target = OUT_DIR / "environment.yml"
        env_target.write_text(
            (HERE / "environment_yml.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        for module_name, target_name in MODULE_MAP.items():
            module_target = OUT_DIR / target_name
            module_target.write_text(
                (HERE / module_name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"wrote {module_target}")
        nb_target = OUT_DIR / NOTEBOOK_NAME
        nbf.write(_notebook(), nb_target)
        nbf.read(nb_target, as_version=4)  # validate round-trip
        print(f"wrote {env_target}")
        print(f"wrote {nb_target}")
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

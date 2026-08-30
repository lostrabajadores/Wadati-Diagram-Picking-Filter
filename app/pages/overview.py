import reflex as rx

from app.components.setup_steps import setup_steps
from app.components.wadati_plot import wadati_plot
from app.states.wadati_state import WadatiState


def _masthead() -> rx.Component:
    return rx.el.header(
        rx.el.div(
            rx.icon("activity", class_name="h-4 w-4 text-[#A6321F]"),
            rx.el.span(
                "Wadati QC · field notebook",
                class_name="text-[11px] tracking-[0.32em] uppercase text-[#5C666F]",
            ),
            class_name="flex items-center gap-3",
        ),
        rx.el.span(
            "step 04 / 04 — embedded panel dashboard · complete",
            class_name="text-[11px] tracking-[0.22em] uppercase text-[#8a7f68]",
        ),
        class_name="w-full flex items-center justify-between border-b border-[#2B2F33]/25 pb-4",
    )


def _stat(label: str, value: str, tone: str) -> rx.Component:
    return rx.el.div(
        rx.el.span(
            label,
            class_name="block text-[10px] tracking-[0.24em] uppercase text-[#8a7f68]",
        ),
        rx.el.span(value, class_name=tone),
        class_name="pr-6",
    )


def _readout() -> rx.Component:
    return rx.el.div(
        _stat(
            "Vp / Vs",
            f"{WadatiState.vp_vs:.3f}",
            "block text-2xl font-semibold text-[#2B2F33] tabular-nums mt-1",
        ),
        _stat(
            "correlation r",
            f"{WadatiState.correlation:.4f}",
            "block text-2xl font-semibold text-[#0E6B6B] tabular-nums mt-1",
        ),
        _stat(
            "retained picks",
            f"{WadatiState.retained} / {WadatiState.total}",
            "block text-2xl font-semibold text-[#2B2F33] tabular-nums mt-1",
        ),
        _stat(
            "rejected",
            WadatiState.rejected_stations,
            "block text-2xl font-semibold text-[#A6321F] mt-1",
        ),
        class_name="grid grid-cols-2 md:grid-cols-4 gap-y-5 border-t border-[#CBBFA6] pt-4 mt-4",
    )


def _legend() -> rx.Component:
    return rx.el.div(
        rx.el.span(
            rx.el.span(
                class_name="inline-block h-2 w-2 rounded-full bg-[#0E6B6B] mr-2"
            ),
            "retained",
            class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#5C666F]",
        ),
        rx.el.span(
            rx.icon("x", class_name="h-3 w-3 text-[#A6321F] mr-1.5"),
            "rejected outlier",
            class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#5C666F]",
        ),
        rx.el.span(
            rx.el.span(class_name="inline-block h-px w-6 bg-[#0E6B6B] mr-2"),
            "least-squares fit",
            class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#5C666F]",
        ),
        class_name="flex flex-wrap items-center gap-x-7 gap-y-2",
    )


def _centerpiece() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.div(
                rx.el.h2(
                    "Wadati diagram",
                    class_name="text-lg font-semibold text-[#2B2F33] tracking-tight",
                ),
                rx.el.p(
                    f"Sample event {WadatiState.event_id} · 12 station pairs",
                    class_name="text-[12px] text-[#5C666F] mt-0.5",
                ),
            ),
            _legend(),
            class_name="flex flex-wrap items-end justify-between gap-4 mb-2",
        ),
        wadati_plot(),
        _readout(),
        rx.el.p(
            "S−P interval against P travel time. In a homogeneous medium the slope equals Vp/Vs − 1, "
            "so a degraded correlation is direct evidence of a mis-identified phase pick. "
            "The seismic-red crosses are the outliers the exhaustive subset search removes "
            "before the fit is accepted; the notebook reproduces this figure with Matplotlib "
            "inside Panel.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-4 max-w-2xl",
        ),
        class_name="flex-1 min-w-0 border border-[#CBBFA6] bg-[#FBF7EF] p-6 md:p-8",
    )


def _aside() -> rx.Component:
    return rx.el.aside(
        rx.el.h2(
            "Run it locally",
            class_name="text-lg font-semibold text-[#2B2F33] tracking-tight",
        ),
        rx.el.p(
            "This page is the companion overview. Jupyter runs on your machine — nothing here executes a local kernel.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-1",
        ),
        setup_steps(),
        rx.el.div(
            rx.el.div(
                rx.icon("file-code-2", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/wadati_qc.ipynb",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("package", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/environment.yml",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("database", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/wadati_ingest.py",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("radar", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/wadati_discovery.py",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("sigma", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/wadati_subset.py",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("flask-conical", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/wadati_subset_examples.py",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon(
                    "layout-dashboard", class_name="h-4 w-4 text-[#0E6B6B]"
                ),
                rx.el.code(
                    "notebooks/wadati_dashboard.py",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("folder", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "notebooks/data/",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            rx.el.div(
                rx.icon("book-open", class_name="h-4 w-4 text-[#0E6B6B]"),
                rx.el.code(
                    "app/notebook/README.md",
                    class_name="text-[12px] font-mono text-[#2B2F33]",
                ),
                class_name="flex items-center gap-2",
            ),
            class_name="flex flex-col gap-2 border-t border-[#CBBFA6] pt-4 mt-1",
        ),
        class_name="w-full lg:w-[26rem] shrink-0 border border-[#CBBFA6] bg-[#F2EBDD]/70 p-6 md:p-7",
    )


def _schema_row(name: str, requirement: str, meaning: str) -> rx.Component:
    return rx.el.div(
        rx.el.code(
            name,
            class_name="text-[12px] font-mono text-[#2B2F33] w-40 shrink-0",
        ),
        rx.el.span(
            requirement,
            class_name="text-[10px] tracking-[0.2em] uppercase text-[#0E6B6B] w-24 shrink-0",
        ),
        rx.el.span(
            meaning, class_name="text-[12.5px] text-[#5C666F] leading-relaxed"
        ),
        class_name="flex items-baseline gap-3 py-1 border-b border-dashed border-[#CBBFA6]/70",
    )


def _code_chip(text: str, tone: str) -> rx.Component:
    return rx.el.code(text, class_name=tone)


def _schemas() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.span(
                "ingestion · canonical schema",
                class_name="text-[10px] tracking-[0.3em] uppercase text-[#8a7f68]",
            ),
            rx.el.span(
                rx.icon("check", class_name="h-3.5 w-3.5 mr-1.5"),
                "validated · step 02 complete",
                class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#0E6B6B]",
            ),
            class_name="flex flex-wrap items-center justify-between gap-3",
        ),
        rx.el.p(
            "Both input families normalize onto one row per event / station P–S pair. Absolute "
            "timestamps or numeric travel times are accepted wherever scientifically valid.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-2 max-w-3xl",
        ),
        rx.el.div(
            rx.el.div(
                _schema_row(
                    "event_id", "required", "catalog / origin identifier"
                ),
                _schema_row("station_id", "required", "station code, e.g. BRG"),
                _schema_row(
                    "network, channel", "optional", "SEED codes when available"
                ),
                _schema_row(
                    "source", "required", "provenance, e.g. csv:picks.csv"
                ),
                _schema_row(
                    "origin_time", "optional", "event origin time, ISO-8601 UTC"
                ),
                _schema_row(
                    "p_time, s_time",
                    "optional",
                    "absolute pick times, ISO-8601 UTC",
                ),
                _schema_row(
                    "p_travel_time",
                    "required",
                    "origin-relative P travel time (s)",
                ),
                _schema_row(
                    "s_minus_p",
                    "required",
                    "S−P interval (s), strictly positive",
                ),
                class_name="flex-1 min-w-0",
            ),
            rx.el.div(
                rx.el.h3(
                    "CSV phase-pick tables",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.p(
                    "Long layout, one row per phase: ",
                    _code_chip(
                        "event_id, station_id, phase, time | travel_time",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Wide layout, one row per pair: ",
                    _code_chip(
                        "p_time, s_time",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    " or ",
                    _code_chip(
                        "p_travel_time, s_minus_p",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.h3(
                    "ObsPy QuakeML catalogs",
                    class_name="text-[13px] font-semibold text-[#2B2F33] mt-4",
                ),
                rx.el.p(
                    "Picks paired per event and station through ",
                    _code_chip(
                        "pick.waveform_id",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    ", phase from ",
                    _code_chip(
                        "phase_hint", "text-[11.5px] font-mono text-[#2B2F33]"
                    ),
                    " with the origin arrival phase as fallback, P travel time against the "
                    "preferred origin — then the first origin that carries a time.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.h3(
                    "Validation states",
                    class_name="text-[13px] font-semibold text-[#2B2F33] mt-4",
                ),
                rx.el.p(
                    "duplicate_phase · missing_p · missing_s · s_not_after_p · malformed_time · "
                    "missing_origin_time · origin_time_conflict · insufficient_pairs",
                    class_name="text-[11.5px] font-mono text-[#A6321F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Every message names the event, the station and the fix; clean input ingests "
                    "silently.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                class_name="flex-1 min-w-0 lg:border-l lg:border-[#CBBFA6] lg:pl-8",
            ),
            class_name="flex flex-col lg:flex-row gap-6 lg:gap-8 mt-4",
        ),
        class_name="w-full border border-[#CBBFA6] bg-[#F2EBDD]/50 p-6 md:p-8 mt-6",
    )


def _discovery_field(label: str, value: str) -> rx.Component:
    return rx.el.div(
        rx.el.span(
            label,
            class_name="block text-[10px] tracking-[0.24em] uppercase text-[#8a7f68]",
        ),
        rx.el.span(
            value,
            class_name="block text-[12.5px] font-mono text-[#2B2F33] mt-1 leading-relaxed",
        ),
        class_name="pr-6",
    )


def _discovery() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.span(
                "discovery · real fdsn data",
                class_name="text-[10px] tracking-[0.3em] uppercase text-[#8a7f68]",
            ),
            rx.el.span(
                rx.icon("radar", class_name="h-3.5 w-3.5 mr-1.5"),
                "button-driven · never on import",
                class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#0E6B6B]",
            ),
            class_name="flex flex-wrap items-center justify-between gap-3",
        ),
        rx.el.p(
            "Hispaniola is the Caribbean island shared by the Dominican Republic and Haiti. "
            "Network LO is the Observatorio Sismólogico Politécnico Loyola (OSPL) network of the "
            "Dominican Republic, and EARTHSCOPE is the only FDSN node that serves it; regional "
            "earthquakes are queried from USGS.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-2 max-w-3xl",
        ),
        rx.el.div(
            _discovery_field(
                "station metadata", "EARTHSCOPE · network LO (OSPL)"
            ),
            _discovery_field("event catalog", "USGS · regional"),
            _discovery_field("default region", "16.5–20.5 N · 75.5–67.0 W"),
            _discovery_field("defaults", "M ≥ 3.0 · configurable UTC range"),
            class_name="grid grid-cols-2 md:grid-cols-4 gap-y-5 border-t border-[#CBBFA6] pt-4 mt-4",
        ),
        rx.el.div(
            rx.el.div(
                rx.el.h3(
                    "Tidy tables",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.p(
                    "Stations: ",
                    _code_chip(
                        "station_uid, network, station_id, site, latitude, longitude, "
                        "elevation_m, start_date, end_date, channels",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Events: ",
                    _code_chip(
                        "event_uid, event_id, origin_time, latitude, longitude, depth_km, "
                        "magnitude, magnitude_type, event_type, region",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-2",
                ),
                rx.el.p(
                    "Panel MultiChoice and MultiSelect controls are populated from the real "
                    "inventory and catalog — no selection is fabricated, and nothing is "
                    "queried until the button is pressed.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-2",
                ),
                class_name="flex-1 min-w-0",
            ),
            rx.el.div(
                rx.el.h3(
                    "Failure states",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.p(
                    "no_data · malformed_range · bad_request · provider_unreachable · "
                    "provider_failure · offline · obspy_missing",
                    class_name="text-[11.5px] font-mono text-[#A6321F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Every code comes back on the returned result with the fix spelled out — "
                    "reversed or unparsable dates, an implausible magnitude threshold, an "
                    "empty region, a rejected request, a provider outage and a machine with "
                    "no network are all reported instead of raising.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Ingestion result API: ",
                    _code_chip(
                        "event_summary() · station_summary() · to_dataframe() · report()",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-4",
                ),
                class_name="flex-1 min-w-0 lg:border-l lg:border-[#CBBFA6] lg:pl-8",
            ),
            class_name="flex flex-col lg:flex-row gap-6 lg:gap-8 mt-4",
        ),
        class_name="w-full border border-[#CBBFA6] bg-[#FBF7EF] p-6 md:p-8 mt-6",
    )


def _step_line(index: str, text: str) -> rx.Component:
    return rx.el.li(
        rx.el.span(
            index,
            class_name="text-[10px] tracking-[0.28em] text-[#8a7f68] font-semibold shrink-0 pt-1",
        ),
        rx.el.span(
            text, class_name="text-[12.5px] text-[#5C666F] leading-relaxed"
        ),
        class_name="flex gap-4 py-1.5 border-b border-dashed border-[#CBBFA6]/70",
    )


def _reason_row(code: str, tone: str, meaning: str) -> rx.Component:
    return rx.el.div(
        rx.el.code(code, class_name=tone),
        rx.el.span(
            meaning, class_name="text-[12.5px] text-[#5C666F] leading-relaxed"
        ),
        class_name="flex items-baseline gap-3 py-1 border-b border-dashed border-[#CBBFA6]/70",
    )


def _subset() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.span(
                "subset search · exhaustive",
                class_name="text-[10px] tracking-[0.3em] uppercase text-[#8a7f68]",
            ),
            rx.el.span(
                rx.icon("check", class_name="h-3.5 w-3.5 mr-1.5"),
                "deterministic · step 03 complete",
                class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#0E6B6B]",
            ),
            class_name="flex flex-wrap items-center justify-between gap-3",
        ),
        rx.el.p(
            "S−P is regressed against P travel time with scipy.stats.linregress over every valid "
            "station pair. When the full set already satisfies the criteria nothing is removed; "
            "otherwise subsets are enumerated one removal at a time, larger subsets first.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-2 max-w-3xl",
        ),
        rx.el.div(
            _discovery_field("min correlation", "|r| ≥ 0.90 · configurable"),
            _discovery_field("min stations", "4 · clamped to a floor of 3"),
            _discovery_field("Vp/Vs bounds", "1.50 … 2.10 · optional"),
            _discovery_field("tie-break", "max |r| → min rmse → station order"),
            class_name="grid grid-cols-2 md:grid-cols-4 gap-y-5 border-t border-[#CBBFA6] pt-4 mt-4",
        ),
        rx.el.div(
            rx.el.div(
                rx.el.h3(
                    "The algorithm, per event",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.ol(
                    _step_line(
                        "01",
                        "Fit every valid station pair; slope + 1 is the Vp/Vs estimate.",
                    ),
                    _step_line(
                        "02",
                        "Correlation, station count and Vp/Vs bounds all met → accept all picks.",
                    ),
                    _step_line(
                        "03",
                        "Otherwise remove one point, then two, then three — larger subsets always first.",
                    ),
                    _step_line(
                        "04",
                        "Stop at the first removal depth that yields qualifying candidates.",
                    ),
                    _step_line(
                        "05",
                        "Choose highest |r|, then lowest residual error, then stable station ordering.",
                    ),
                    _step_line(
                        "06",
                        "Reject when no qualifying subset remains at or above the minimum stations.",
                    ),
                    class_name="flex flex-col mt-2",
                ),
                rx.el.h3(
                    "Reported fields",
                    class_name="text-[13px] font-semibold text-[#2B2F33] mt-4",
                ),
                rx.el.p(
                    "status · reason · original_stations · retained_stations · removed_stations · "
                    "search_depth · combinations_evaluated · slope · intercept · pearson_r · "
                    "r_squared · p_value · slope_stderr · intercept_stderr · rmse · mae · "
                    "max_abs_residual · residual_std · vp_vs · vp_vs_stderr · min_correlation · "
                    "min_stations · vp_vs_min · vp_vs_max",
                    class_name="text-[11.5px] font-mono text-[#2B2F33] leading-relaxed mt-1",
                ),
                class_name="flex-1 min-w-0",
            ),
            rx.el.div(
                rx.el.h3(
                    "Acceptance and rejection states",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.div(
                    _reason_row(
                        "accepted_all_picks",
                        "text-[11.5px] font-mono text-[#0E6B6B] w-52 shrink-0",
                        "the full set already qualifies; nothing removed",
                    ),
                    _reason_row(
                        "accepted_after_removal",
                        "text-[11.5px] font-mono text-[#0E6B6B] w-52 shrink-0",
                        "a qualifying subset was found at the shallowest depth",
                    ),
                    _reason_row(
                        "rejected_correlation",
                        "text-[11.5px] font-mono text-[#A6321F] w-52 shrink-0",
                        "no subset ever reached the threshold — the picks disagree",
                    ),
                    _reason_row(
                        "rejected_vp_vs",
                        "text-[11.5px] font-mono text-[#A6321F] w-52 shrink-0",
                        "correlation was satisfied, but the velocity ratio is anomalous",
                    ),
                    _reason_row(
                        "rejected_insufficient_stations",
                        "text-[11.5px] font-mono text-[#A6321F] w-52 shrink-0",
                        "fewer valid P–S pairs than the minimum station count",
                    ),
                    _reason_row(
                        "rejected_search_truncated",
                        "text-[11.5px] font-mono text-[#A6321F] w-52 shrink-0",
                        "the combination budget was exhausted before a subset qualified",
                    ),
                    class_name="mt-2",
                ),
                rx.el.h3(
                    "Exports and examples",
                    class_name="text-[13px] font-semibold text-[#2B2F33] mt-4",
                ),
                rx.el.p(
                    "Dataset-wide runs consume canonical ingestion pairs: ",
                    _code_chip(
                        "run_subset_search(result, SearchConfig()) → DatasetQC",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    ". Two real CSV exports — detailed per-event QC and the per-pick "
                    "retained / outlier table — are written to disk and offered as Panel "
                    "FileDownload widgets.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "all_pass · one_outlier · two_outliers · anomalous_vp_vs · irrecoverable",
                    class_name="text-[11.5px] font-mono text-[#0E6B6B] leading-relaxed mt-3",
                ),
                rx.el.p(
                    "Five deterministic worked examples assert the expected status, reason, "
                    "search depth and removed stations — run them with ",
                    _code_chip(
                        "python -m app.notebook.subset_examples",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    ".",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                class_name="flex-1 min-w-0 lg:border-l lg:border-[#CBBFA6] lg:pl-8",
            ),
            class_name="flex flex-col lg:flex-row gap-6 lg:gap-8 mt-4",
        ),
        class_name="w-full border border-[#CBBFA6] bg-[#F2EBDD]/50 p-6 md:p-8 mt-6",
    )


def _dashboard() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.span(
                "dashboard · embedded panel",
                class_name="text-[10px] tracking-[0.3em] uppercase text-[#8a7f68]",
            ),
            rx.el.span(
                rx.icon("check", class_name="h-3.5 w-3.5 mr-1.5"),
                "workflow complete · step 04",
                class_name="flex items-center text-[11px] tracking-[0.16em] uppercase text-[#0E6B6B]",
            ),
            class_name="flex flex-wrap items-center justify-between gap-3",
        ),
        rx.el.p(
            "The notebook ends in a dense field-notebook workspace built with Panel and centred "
            "on the real Wadati plot: event-by-event navigation with status filtering, a linked "
            "editable pick table, manual re-picking with validation and provenance, a compared "
            "second pass, and optional CPU SeisBench assistance.",
            class_name="text-[13px] text-[#5C666F] leading-relaxed mt-2 max-w-3xl",
        ),
        rx.el.div(
            _discovery_field(
                "navigation", "◀ ▶ · all / accepted / rejected / edited"
            ),
            _discovery_field(
                "editing", "validate → stage → apply · undo · reset"
            ),
            _discovery_field("second pass", "one event or the whole dataset"),
            _discovery_field(
                "exports", "original + revised · event + pick CSV"
            ),
            class_name="grid grid-cols-2 md:grid-cols-4 gap-y-5 border-t border-[#CBBFA6] pt-4 mt-4",
        ),
        rx.el.div(
            rx.el.div(
                rx.el.h3(
                    "Manual re-picking",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.p(
                    "P and S times are edited as numeric travel times or absolute ISO "
                    "timestamps. Every edit is validated first — the P travel time and S−P must "
                    "be finite and strictly positive, absolute times need the origin time, and "
                    "an edit that changes nothing is refused. Edits are staged, then applied, "
                    "and each one is kept as a provenance record that can be undone or reset.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Session API: ",
                    _code_chip(
                        "stage_edit() · apply_pending() · undo_last() · reset_event() · "
                        "rerun_event() · rerun_all() · comparison_rows() · export_csv()",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-3",
                ),
                rx.el.p(
                    "Retained picks are teal, removed outliers seismic red, and revised picks "
                    "amber diamonds — original and revised metrics and dataset counts sit side "
                    "by side, with all four QC CSVs downloadable.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-3",
                ),
                class_name="flex-1 min-w-0",
            ),
            rx.el.div(
                rx.el.h3(
                    "Optional SeisBench assist · CPU, lazy",
                    class_name="text-[13px] font-semibold text-[#2B2F33]",
                ),
                rx.el.p(
                    "PhaseNet · EQTransformer · GPD",
                    class_name="text-[11.5px] font-mono text-[#0E6B6B] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "Uploaded or local MiniSEED is read with ObsPy, the chosen model class and "
                    "pretrained weight set come from the real ",
                    _code_chip(
                        "list_pretrained()",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    " listing, and inference runs on the CPU through the official ",
                    _code_chip(
                        "model.classify(stream)",
                        "text-[11.5px] font-mono text-[#2B2F33]",
                    ),
                    " API. Candidate P and S picks are shown with their real confidence and "
                    "reach a pick only after an explicit selection — nothing is overwritten "
                    "silently and no prediction is ever mocked.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.p(
                    "seisbench_missing · torch_missing · unsupported_model · unknown_weights · "
                    "weights_unavailable · offline · inference_failed · read_failed",
                    class_name="text-[11.5px] font-mono text-[#A6321F] leading-relaxed mt-3",
                ),
                rx.el.p(
                    "Weights download to ~/.seisbench/models on first use; an offline machine "
                    "with an empty cache is reported truthfully. Nothing — network, model or "
                    "weights — is touched at notebook import.",
                    class_name="text-[12.5px] text-[#5C666F] leading-relaxed mt-1",
                ),
                class_name="flex-1 min-w-0 lg:border-l lg:border-[#CBBFA6] lg:pl-8",
            ),
            class_name="flex flex-col lg:flex-row gap-6 lg:gap-8 mt-4",
        ),
        class_name="w-full border border-[#CBBFA6] bg-[#FBF7EF] p-6 md:p-8 mt-6",
    )


def _roadmap() -> rx.Component:
    return rx.el.section(
        rx.el.span(
            "roadmap",
            class_name="text-[10px] tracking-[0.3em] uppercase text-[#8a7f68]",
        ),
        rx.el.div(
            rx.el.p(
                rx.el.strong(
                    "01 Foundation · done", class_name="text-[#0E6B6B]"
                ),
                " — conda-forge environment, Panel-initialised notebook, theme and Wadati centerpiece.",
                class_name="text-[13px] text-[#5C666F] leading-relaxed",
            ),
            rx.el.p(
                rx.el.strong(
                    "02 Ingestion · done", class_name="text-[#0E6B6B]"
                ),
                " — canonical schema, CSV and QuakeML loaders, real LO / OSPL and Hispaniola FDSN discovery, actionable errors.",
                class_name="text-[13px] text-[#5C666F] leading-relaxed",
            ),
            rx.el.p(
                rx.el.strong(
                    "03 Subset search · done", class_name="text-[#0E6B6B]"
                ),
                " — exhaustive removal search with min correlation 0.9, min stations (floor 3), "
                "Vp/Vs bounds, explicit rejection reasons and downloadable QC results.",
                class_name="text-[13px] text-[#5C666F] leading-relaxed",
            ),
            rx.el.p(
                rx.el.strong(
                    "04 Dashboard · done", class_name="text-[#0E6B6B]"
                ),
                " — embedded Panel workspace: event-by-event inspection and filtering, manual "
                "re-picking with provenance and undo, optional CPU SeisBench assist, and a "
                "compared second pass with downloadable QC exports.",
                class_name="text-[13px] text-[#5C666F] leading-relaxed",
            ),
            class_name="grid md:grid-cols-2 gap-x-10 gap-y-3 mt-3",
        ),
        class_name="w-full border-t border-[#2B2F33]/20 pt-6 mt-10",
    )


def overview() -> rx.Component:
    return rx.el.main(
        rx.el.div(
            _masthead(),
            rx.el.div(
                rx.el.h1(
                    "Interactive Wadati quality control",
                    class_name="text-3xl md:text-[2.6rem] leading-[1.1] font-semibold text-[#2B2F33] tracking-tight max-w-3xl",
                ),
                rx.el.p(
                    "A complete, reproducible Python 3.11 notebook for auditing P and S phase picks, "
                    "estimating Vp/Vs, re-picking the outliers by hand or with SeisBench, and "
                    "comparing the revised second pass against the original run.",
                    class_name="text-[15px] text-[#5C666F] leading-relaxed mt-4 max-w-2xl",
                ),
                class_name="pt-9 pb-8",
            ),
            rx.el.div(
                _centerpiece(),
                _aside(),
                class_name="flex flex-col lg:flex-row gap-6 w-full items-stretch",
            ),
            _schemas(),
            _discovery(),
            _subset(),
            _dashboard(),
            _roadmap(),
            rx.el.footer(
                rx.el.span(
                    "obspy · panel · scipy · seisbench (cpu)",
                    class_name="text-[11px] tracking-[0.2em] uppercase text-[#8a7f68]",
                ),
                class_name="w-full pt-6 mt-6 border-t border-[#CBBFA6]",
            ),
            class_name="mx-auto w-full max-w-6xl px-6 md:px-10 pb-16",
        ),
        class_name="min-h-screen w-full bg-[#F6F1E7] font-['Inter'] antialiased",
    )

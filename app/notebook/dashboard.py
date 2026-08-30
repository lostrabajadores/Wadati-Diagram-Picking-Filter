"""Embedded Panel dashboard for the Wadati quality-control workflow — step 4.

Two clearly separated layers live here:

**Logic** (pure Python, no Panel, fully testable)
    :class:`DashboardSession` holds the first-pass :class:`DatasetQC`, the
    editable working picks, the provenance-carrying edit log, and the
    second-pass rerun of a single event or of the whole dataset.  Every edit is
    validated, staged, applied, undone or reset explicitly — nothing is ever
    silently overwritten.

**Optional waveform assistance** (SeisBench, CPU only, lazy)
    :func:`available_weights` and :func:`load_picker` touch SeisBench/torch
    **only when called**, so importing this module never downloads weights and
    never touches the network.  :func:`repick_stream` runs the official
    ``model.classify(stream)`` API and returns real candidate picks with their
    confidence; there are no mock predictions anywhere.  Missing packages, an
    unknown weight name, an empty cache and an offline machine are reported as
    truthful, actionable messages.

**Panel layer**
    :func:`build_dashboard` assembles the dense, asymmetric field-notebook
    workspace centred on the real Wadati plot: warm mineral paper, charcoal and
    slate type, thin geological rule lines, deep **teal** retained/accepted,
    **seismic red** outlier/rejected, and **amber** for revised picks.

Run the offline self-check::

    python -m app.notebook.dashboard
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # in-app import
    from app.notebook.ingest import format_timestamp, parse_timestamp
    from app.notebook.subset_search import (
        STATUS_ACCEPTED,
        DatasetQC,
        EventQC,
        SearchConfig,
        WadatiPoint,
        event_qc_csv_text,
        fit_subset,
        pick_qc_csv_text,
        run_subset_search,
        search_event,
    )
except ImportError:  # notebook-local copies
    from wadati_ingest import format_timestamp, parse_timestamp  # type: ignore
    from wadati_subset import (  # type: ignore[no-redef]
        STATUS_ACCEPTED,
        DatasetQC,
        EventQC,
        SearchConfig,
        WadatiPoint,
        event_qc_csv_text,
        fit_subset,
        pick_qc_csv_text,
        run_subset_search,
        search_event,
    )

# ---------------------------------------------------------------------------
# palette — the established field-notebook identity, plus amber for revisions
# ---------------------------------------------------------------------------
THEME: dict[str, str] = {
    "paper": "#F6F1E7",
    "paper_deep": "#EDE5D6",
    "rule": "#CBBFA6",
    "ink": "#2B2F33",
    "slate": "#5C666F",
    "teal": "#0E6B6B",
    "red": "#A6321F",
    "amber": "#B57415",
}

#: SeisBench model classes this dashboard supports (all real phase pickers).
SUPPORTED_MODELS: tuple[str, ...] = ("PhaseNet", "EQTransformer", "GPD")

DASHBOARD_DOC = """\
WADATI DASHBOARD — step 4
-------------------------
DashboardSession(ingest_result, config)
  .original                first-pass DatasetQC (never mutated)
  .event_ids()             events in catalog order
  .filtered_event_ids(f)   f in {all, accepted, rejected, edited, revised}
  .current(event_id)       revised EventQC when it exists, else the original
  .picks(event_id)         editable picks (revised picks carry provenance)
  .pick_table(event_id)    row dicts for the editable table
  .stage_edit(...)         validated, provenance-carrying pending edit
  .apply_pending(...)      commit staged edits onto the working picks
  .discard_pending(...)    drop staged edits without applying them
  .undo_last(event_id)     revert the most recent applied edit
  .reset_event(event_id)   drop every edit for one event
  .reset_all()             back to the ingested picks
  .rerun_event(event_id)   second pass for one event
  .rerun_all()             second pass for the whole dataset
  .comparison_rows(id)     original vs revised metrics, side by side
  .dataset_comparison()    original vs revised dataset counts
  .export_csv(which, kind) 'original'/'revised' x 'events'/'picks' -> CSV text

EDIT VALIDATION
  p_travel_time must be finite and > 0
  s_minus_p     must be finite and > 0  (S strictly after P)
  absolute p_time / s_time need the event origin time
  an edit that changes nothing is rejected

SEISBENCH (optional, CPU, lazy)
  available_weights(model)          real list_pretrained() names
  load_picker(model, weights)       cached, eval() mode, CPU
  read_mseed(bytes | path)          ObsPy MiniSEED reader
  repick_stream(stream, m, w)       official model.classify(stream) API
  error codes: seisbench_missing, torch_missing, unsupported_model,
               unknown_weights, weights_unavailable, offline, inference_failed,
               read_failed
"""


class PickEditError(ValueError):
    """A manual or model-assisted edit that failed validation."""


# ---------------------------------------------------------------------------
# editable picks and provenance
# ---------------------------------------------------------------------------
@dataclass
class EditablePick:
    """One station's P/S pair, editable and provenance-aware."""

    event_id: str
    station_id: str
    network: str = ""
    channel: str = ""
    origin_time: float | None = None
    p_travel_time: float = 0.0
    s_minus_p: float = 0.0
    original_p_travel_time: float = 0.0
    original_s_minus_p: float = 0.0
    provenance: str = "ingested"
    note: str = ""

    @property
    def revised(self) -> bool:
        return (
            abs(self.p_travel_time - self.original_p_travel_time) > 1e-9
            or abs(self.s_minus_p - self.original_s_minus_p) > 1e-9
        )

    @property
    def p_time(self) -> float | None:
        if self.origin_time is None:
            return None
        return self.origin_time + self.p_travel_time

    @property
    def s_time(self) -> float | None:
        if self.origin_time is None:
            return None
        return self.origin_time + self.p_travel_time + self.s_minus_p

    @property
    def apparent_vp_vs(self) -> float:
        if self.p_travel_time <= 0.0:
            return float("nan")
        return 1.0 + self.s_minus_p / self.p_travel_time

    def as_point(self) -> WadatiPoint:
        return WadatiPoint(
            station_id=self.station_id,
            p_travel_time=float(self.p_travel_time),
            s_minus_p=float(self.s_minus_p),
            network=self.network,
            channel=self.channel,
        )


@dataclass(frozen=True)
class PickEdit:
    """An audited change to one pick — the provenance record."""

    event_id: str
    station_id: str
    old_p_travel_time: float
    old_s_minus_p: float
    new_p_travel_time: float
    new_s_minus_p: float
    provenance: str = "manual"
    note: str = ""
    staged_at: str = ""

    @property
    def delta_p(self) -> float:
        return self.new_p_travel_time - self.old_p_travel_time

    @property
    def delta_s_minus_p(self) -> float:
        return self.new_s_minus_p - self.old_s_minus_p

    def describe(self) -> str:
        return (
            f"{self.event_id} · {self.station_id}: "
            f"t_P {self.old_p_travel_time:.3f} → {self.new_p_travel_time:.3f} s, "
            f"S−P {self.old_s_minus_p:.3f} → {self.new_s_minus_p:.3f} s "
            f"[{self.provenance}]" + (f" — {self.note}" if self.note else "")
        )

    def as_row(self) -> dict[str, str | float]:
        return {
            "event_id": self.event_id,
            "station_id": self.station_id,
            "provenance": self.provenance,
            "note": self.note,
            "staged_at": self.staged_at,
            "old_p_travel_time": round(self.old_p_travel_time, 4),
            "new_p_travel_time": round(self.new_p_travel_time, 4),
            "delta_p": round(self.delta_p, 4),
            "old_s_minus_p": round(self.old_s_minus_p, 4),
            "new_s_minus_p": round(self.new_s_minus_p, 4),
            "delta_s_minus_p": round(self.delta_s_minus_p, 4),
        }


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PickEditError(f"{label}: {value!r} is not a number") from exc
    if not math.isfinite(number):
        raise PickEditError(f"{label}: {number} is not finite")
    return number


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
FILTERS: tuple[str, ...] = ("all", "accepted", "rejected", "edited", "revised")


@dataclass
class DashboardSession:
    """First pass, editable working picks, and the second pass."""

    ingest_result: object
    config: SearchConfig = field(default_factory=SearchConfig)
    original: DatasetQC = field(init=False)
    working: dict[str, list[EditablePick]] = field(
        init=False, default_factory=dict
    )
    applied: list[PickEdit] = field(init=False, default_factory=list)
    pending: list[PickEdit] = field(init=False, default_factory=list)
    revised_events: dict[str, EventQC] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.config = self.config.normalized()[0]
        self.original = run_subset_search(self.ingest_result, self.config)
        self.source = getattr(self.ingest_result, "source", "<ingest>")
        by_event: dict[str, dict[str, EditablePick]] = {}
        for pair in getattr(self.ingest_result, "usable_pairs", []):
            by_event.setdefault(pair.event_id, {})[pair.station_id] = (
                EditablePick(
                    event_id=str(pair.event_id),
                    station_id=str(pair.station_id),
                    network=str(getattr(pair, "network", "") or ""),
                    channel=str(getattr(pair, "channel", "") or ""),
                    origin_time=getattr(pair, "origin_time", None),
                    p_travel_time=float(pair.p_travel_time),
                    s_minus_p=float(pair.s_minus_p),
                    original_p_travel_time=float(pair.p_travel_time),
                    original_s_minus_p=float(pair.s_minus_p),
                )
            )
        for result in self.original.results:
            picks = by_event.get(result.event_id, {})
            ordered = sorted(
                picks.values(), key=lambda p: (p.p_travel_time, p.station_id)
            )
            self.working[result.event_id] = ordered

    # --- navigation ------------------------------------------------------
    def event_ids(self) -> list[str]:
        return [r.event_id for r in self.original.results]

    def current(self, event_id: str) -> EventQC | None:
        revised = self.revised_events.get(event_id)
        if revised is not None:
            return revised
        return self.original.by_id(event_id)

    def status(self, event_id: str) -> str:
        result = self.current(event_id)
        return result.status if result is not None else "unknown"

    def edited_event_ids(self) -> list[str]:
        return [
            event_id
            for event_id in self.event_ids()
            if any(p.revised for p in self.working.get(event_id, []))
        ]

    def filtered_event_ids(self, which: str = "all") -> list[str]:
        which = (which or "all").lower()
        if which == "accepted":
            return [
                e for e in self.event_ids() if self.status(e) == STATUS_ACCEPTED
            ]
        if which == "rejected":
            return [
                e for e in self.event_ids() if self.status(e) != STATUS_ACCEPTED
            ]
        if which == "edited":
            return self.edited_event_ids()
        if which == "revised":
            return [e for e in self.event_ids() if e in self.revised_events]
        return self.event_ids()

    def neighbour(self, event_id: str, step: int, which: str = "all") -> str:
        """Previous / next event inside the active filter (wrapping)."""
        ids = self.filtered_event_ids(which) or self.event_ids()
        if not ids:
            return event_id
        if event_id not in ids:
            return ids[0]
        return ids[(ids.index(event_id) + step) % len(ids)]

    # --- picks -----------------------------------------------------------
    def picks(self, event_id: str) -> list[EditablePick]:
        return self.working.get(event_id, [])

    def pick(self, event_id: str, station_id: str) -> EditablePick:
        for candidate in self.picks(event_id):
            if candidate.station_id == station_id:
                return candidate
        raise PickEditError(
            f"station {station_id!r} is not part of event {event_id!r}"
        )

    def points(self, event_id: str) -> list[WadatiPoint]:
        return [p.as_point() for p in self.picks(event_id)]

    def station_ids(self, event_id: str) -> list[str]:
        return [p.station_id for p in self.picks(event_id)]

    def pick_table(self, event_id: str) -> list[dict[str, str | float | bool]]:
        """Editable-table rows carrying state, residual and provenance."""
        result = self.current(event_id)
        outliers = set(result.removed_stations) if result else set()
        fit = None
        if result is not None:
            fit = result.fit if result.fit.valid else result.initial_fit
        rows: list[dict[str, str | float | bool]] = []
        for pick in self.picks(event_id):
            if fit is not None and fit.valid:
                residual = pick.s_minus_p - (
                    fit.intercept + fit.slope * pick.p_travel_time
                )
            else:
                residual = float("nan")
            rows.append(
                {
                    "station_id": pick.station_id,
                    "network": pick.network,
                    "channel": pick.channel,
                    "p_travel_time": round(pick.p_travel_time, 3),
                    "s_minus_p": round(pick.s_minus_p, 3),
                    "p_time": format_timestamp(pick.p_time),
                    "s_time": format_timestamp(pick.s_time),
                    "state": "outlier"
                    if pick.station_id in outliers
                    else "retained",
                    "revised": pick.revised,
                    "residual": round(residual, 3)
                    if math.isfinite(residual)
                    else float("nan"),
                    "apparent_vp_vs": round(pick.apparent_vp_vs, 3),
                    "provenance": pick.provenance,
                }
            )
        return rows

    # --- editing ---------------------------------------------------------
    def stage_edit(
        self,
        event_id: str,
        station_id: str,
        p_travel_time: float | None = None,
        s_minus_p: float | None = None,
        p_time: str | None = None,
        s_time: str | None = None,
        provenance: str = "manual",
        note: str = "",
    ) -> PickEdit:
        """Validate an edit and stage it (nothing is applied yet).

        Either numeric ``p_travel_time`` / ``s_minus_p`` **or** absolute ISO
        ``p_time`` / ``s_time`` may be supplied; absolute times need the event
        origin time.
        """
        pick = self.pick(event_id, station_id)
        new_p = pick.p_travel_time
        new_sp = pick.s_minus_p

        if p_time or s_time:
            if pick.origin_time is None:
                raise PickEditError(
                    "this event carries no origin time, so absolute pick times "
                    "cannot be converted — edit the numeric travel times instead"
                )
            try:
                p_epoch = (
                    parse_timestamp(p_time)
                    if p_time
                    else float(pick.origin_time) + pick.p_travel_time
                )
                s_epoch = (
                    parse_timestamp(s_time)
                    if s_time
                    else float(pick.origin_time)
                    + pick.p_travel_time
                    + pick.s_minus_p
                )
            except ValueError as exc:
                raise PickEditError(f"absolute pick time — {exc}") from exc
            new_p = p_epoch - float(pick.origin_time)
            new_sp = s_epoch - p_epoch
        if p_travel_time is not None:
            new_p = _finite(p_travel_time, "p_travel_time")
        if s_minus_p is not None:
            new_sp = _finite(s_minus_p, "s_minus_p")

        new_p = _finite(new_p, "p_travel_time")
        new_sp = _finite(new_sp, "s_minus_p")
        if new_p <= 0.0:
            raise PickEditError(
                f"p_travel_time must be positive (got {new_p:.3f} s); the P pick "
                "cannot precede the origin time"
            )
        if new_sp <= 0.0:
            raise PickEditError(
                f"S−P must be positive (got {new_sp:.3f} s); S must arrive strictly "
                "after P — the phases look swapped"
            )
        if (
            abs(new_p - pick.p_travel_time) <= 1e-6
            and abs(new_sp - pick.s_minus_p) <= 1e-6
        ):
            raise PickEditError(
                "this edit changes nothing — enter a different P or S time"
            )

        edit = PickEdit(
            event_id=event_id,
            station_id=station_id,
            old_p_travel_time=pick.p_travel_time,
            old_s_minus_p=pick.s_minus_p,
            new_p_travel_time=new_p,
            new_s_minus_p=new_sp,
            provenance=provenance or "manual",
            note=note,
            staged_at=_now(),
        )
        self.pending.append(edit)
        return edit

    def pending_edits(self, event_id: str | None = None) -> list[PickEdit]:
        if event_id is None:
            return list(self.pending)
        return [e for e in self.pending if e.event_id == event_id]

    def applied_edits(self, event_id: str | None = None) -> list[PickEdit]:
        if event_id is None:
            return list(self.applied)
        return [e for e in self.applied if e.event_id == event_id]

    def apply_pending(self, event_id: str | None = None) -> list[PickEdit]:
        """Commit the staged edits onto the working picks."""
        chosen = self.pending_edits(event_id)
        for edit in chosen:
            pick = self.pick(edit.event_id, edit.station_id)
            pick.p_travel_time = edit.new_p_travel_time
            pick.s_minus_p = edit.new_s_minus_p
            pick.provenance = edit.provenance
            pick.note = edit.note
            self.applied.append(edit)
            self.pending.remove(edit)
        for edit in chosen:
            self._resort(edit.event_id)
        return chosen

    def discard_pending(self, event_id: str | None = None) -> list[PickEdit]:
        dropped = self.pending_edits(event_id)
        for edit in dropped:
            self.pending.remove(edit)
        return dropped

    def undo_last(self, event_id: str | None = None) -> PickEdit | None:
        """Revert the most recent applied edit (optionally for one event)."""
        candidates = self.applied_edits(event_id)
        if not candidates:
            return None
        edit = candidates[-1]
        pick = self.pick(edit.event_id, edit.station_id)
        pick.p_travel_time = edit.old_p_travel_time
        pick.s_minus_p = edit.old_s_minus_p
        self.applied.remove(edit)
        remaining = self.applied_edits(edit.event_id)
        station_edits = [
            e for e in remaining if e.station_id == edit.station_id
        ]
        if station_edits:
            pick.provenance = station_edits[-1].provenance
            pick.note = station_edits[-1].note
        else:
            pick.provenance = "ingested"
            pick.note = ""
        self._resort(edit.event_id)
        return edit

    def reset_event(self, event_id: str) -> int:
        """Drop every edit for one event and forget its rerun."""
        count = 0
        for pick in self.picks(event_id):
            if pick.revised:
                count += 1
            pick.p_travel_time = pick.original_p_travel_time
            pick.s_minus_p = pick.original_s_minus_p
            pick.provenance = "ingested"
            pick.note = ""
        self.applied = [e for e in self.applied if e.event_id != event_id]
        self.pending = [e for e in self.pending if e.event_id != event_id]
        self.revised_events.pop(event_id, None)
        self._resort(event_id)
        return count

    def reset_all(self) -> int:
        return sum(self.reset_event(event_id) for event_id in self.event_ids())

    def _resort(self, event_id: str) -> None:
        picks = self.working.get(event_id)
        if picks:
            picks.sort(key=lambda p: (p.p_travel_time, p.station_id))

    # --- second pass -----------------------------------------------------
    def rerun_event(self, event_id: str) -> EventQC:
        """Second pass for one event, using the current working picks."""
        result = search_event(
            event_id,
            self.points(event_id),
            self.config,
            source=f"{self.source} (revised)",
        )
        self.revised_events[event_id] = result
        return result

    def rerun_all(self) -> DatasetQC:
        """Second pass over every event."""
        for event_id in self.event_ids():
            self.rerun_event(event_id)
        return self.revised_dataset()

    def revised_dataset(self) -> DatasetQC:
        """Dataset built from reruns, falling back to the first pass."""
        results: list[EventQC] = []
        for event_id in self.event_ids():
            result = self.revised_events.get(event_id) or self.original.by_id(
                event_id
            )
            if result is not None:
                results.append(result)
        return DatasetQC(
            source=f"{self.source} (revised)",
            config=self.config,
            results=results,
            skipped=list(self.original.skipped),
        )

    @property
    def has_revision(self) -> bool:
        return bool(self.revised_events)

    # --- comparison ------------------------------------------------------
    def comparison_rows(
        self, event_id: str
    ) -> list[dict[str, str | float | int]]:
        """Original vs revised metrics for one event, side by side."""
        original = self.original.by_id(event_id)
        revised = self.revised_events.get(event_id)
        if original is None:
            return []

        def snapshot(result: EventQC | None) -> dict[str, str | float | int]:
            if result is None:
                return {}
            fit = result.fit if result.fit.valid else result.initial_fit
            return {
                "status": result.status,
                "reason": result.reason,
                "retained": f"{result.retained_stations}/{result.original_stations}",
                "outliers": ", ".join(result.outlier_stations) or "—",
                "pearson_r": round(float(fit.pearson_r), 4),
                "r_squared": round(float(fit.r_squared), 4),
                "rmse": round(float(fit.rmse), 4),
                "vp_vs": round(float(fit.vp_vs), 4),
                "vp_vs_stderr": round(float(fit.vp_vs_stderr), 4),
                "search_depth": result.search_depth,
            }

        left = snapshot(original)
        right = snapshot(revised)
        rows: list[dict[str, str | float | int]] = []
        for metric in left:
            rows.append(
                {
                    "metric": metric,
                    "original": left[metric],
                    "revised": right.get(metric, "— not rerun —"),
                }
            )
        return rows

    def dataset_comparison(self) -> list[dict[str, str | float | int]]:
        """Original vs revised dataset counts."""
        left = self.original.summary()
        right = self.revised_dataset().summary()
        keys = [
            "events",
            "accepted",
            "rejected",
            "picks_total",
            "picks_retained",
            "picks_removed",
            "vp_vs_mean",
            "vp_vs_min",
            "vp_vs_max",
        ]
        return [
            {
                "count": key,
                "original": left.get(key, ""),
                "revised": right.get(key, ""),
            }
            for key in keys
        ]

    def edit_log_rows(self) -> list[dict[str, str | float]]:
        return [edit.as_row() for edit in self.applied]

    # --- exports ---------------------------------------------------------
    def export_csv(self, which: str = "revised", kind: str = "events") -> str:
        dataset = (
            self.revised_dataset() if which == "revised" else self.original
        )
        if kind == "picks":
            return pick_qc_csv_text(dataset)
        return event_qc_csv_text(dataset)

    def write_exports(self, directory: str | Path) -> dict[str, Path]:
        """Write all four QC CSVs and return their paths."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for which in ("original", "revised"):
            for kind in ("events", "picks"):
                path = target / f"wadati_{which}_{kind}_qc.csv"
                path.write_text(self.export_csv(which, kind), encoding="utf-8")
                written[f"{which}_{kind}"] = path
        return written

    def report(self) -> str:
        lines = [
            f"source            {self.source}",
            f"criteria          {self.config.describe()}",
            f"events            {len(self.event_ids())}  "
            f"(accepted {len(self.original.accepted)}, "
            f"rejected {len(self.original.rejected)})",
            f"edited events     {len(self.edited_event_ids())}",
            f"applied edits     {len(self.applied)}  "
            f"(pending {len(self.pending)})",
            f"reruns            {len(self.revised_events)}",
        ]
        if self.applied:
            lines.append("")
            lines.extend(f"  {edit.describe()}" for edit in self.applied)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# optional SeisBench assistance — lazy, CPU only, no mock predictions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeisBenchError(Exception):
    """A truthful, actionable SeisBench / waveform failure."""

    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class CandidatePick:
    """One model-proposed pick, with its real confidence."""

    trace_id: str
    station_id: str
    phase: str
    time: float
    confidence: float
    model: str
    weights: str

    @property
    def provenance(self) -> str:
        return f"seisbench:{self.model}/{self.weights}"

    def as_row(self) -> dict[str, str | float]:
        return {
            "trace_id": self.trace_id,
            "station_id": self.station_id,
            "phase": self.phase,
            "time": format_timestamp(self.time),
            "confidence": round(self.confidence, 4),
            "model": self.model,
            "weights": self.weights,
        }


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _model_class(name: str):
    if name not in SUPPORTED_MODELS:
        raise SeisBenchError(
            "unsupported_model",
            f"{name!r} is not one of the supported phase pickers "
            f"({', '.join(SUPPORTED_MODELS)})",
        )
    try:
        import seisbench.models as sbm
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "seisbench_missing",
            f"SeisBench is not importable ({e}); install seisbench>=0.12.5 "
            "together with the CPU build of torch",
        ) from e
    model_class = getattr(sbm, name, None)
    if model_class is None:
        raise SeisBenchError(
            "unsupported_model",
            f"the installed SeisBench does not provide {name}; upgrade seisbench",
        )
    return model_class


def available_weights(model_name: str) -> list[str]:
    """Real ``list_pretrained()`` weight names for one model class.

    Contacts the SeisBench repository, so it only runs when called.  A missing
    package or an offline machine raises :class:`SeisBenchError`.
    """
    model_class = _model_class(model_name)
    try:
        return [str(name) for name in model_class.list_pretrained()]
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "offline",
            f"the pretrained weight list for {model_name} could not be fetched "
            f"({e}); SeisBench needs network access for the listing, or a "
            "pre-populated ~/.seisbench/models cache",
        ) from e


def load_picker(model_name: str, weights: str):
    """Lazily load a pretrained picker onto the **CPU**, in ``eval()`` mode."""
    key = (model_name, weights)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model_class = _model_class(model_name)
    if not str(weights).strip():
        raise SeisBenchError(
            "unknown_weights",
            f"no pretrained weight name was selected for {model_name}",
        )
    try:
        import torch
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "torch_missing",
            f"PyTorch is not importable ({e}); install the CPU build of torch",
        ) from e
    try:
        model = model_class.from_pretrained(weights)
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "weights_unavailable",
            f"{model_name}/{weights} could not be loaded ({e}); the weights are "
            "downloaded on first use to ~/.seisbench/models — with no network "
            "access they must already be cached there, and the name must appear "
            "in list_pretrained()",
        ) from e
    try:
        model.eval()
        if torch.cuda.is_available():  # pragma: no cover - CPU-only runtime
            model.cpu()
    except Exception as e:  # pragma: no cover - defensive
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "weights_unavailable",
            f"{model_name}/{weights} failed to initialise: {e}",
        ) from e
    _MODEL_CACHE[key] = model
    return model


def read_mseed(data: bytes | str | Path):
    """Read uploaded bytes or a local path as MiniSEED with ObsPy."""
    try:
        from obspy import read
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "read_failed", f"ObsPy is required to read MiniSEED ({e})"
        ) from e
    try:
        if isinstance(data, bytes):
            stream = read(io.BytesIO(data), format="MSEED")
        else:
            target = Path(data)
            if not target.exists():
                raise SeisBenchError(
                    "read_failed",
                    f"{target} does not exist — check the path, or upload the file",
                )
            stream = read(target, format="MSEED")
    except SeisBenchError:
        logging.exception("Unexpected error")
        raise
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "read_failed",
            f"the waveform could not be read as MiniSEED ({e}); ObsPy needs a "
            "valid .mseed / .msd file",
        ) from e
    if len(stream) == 0:
        raise SeisBenchError(
            "read_failed", "the MiniSEED file contains no traces"
        )
    return stream


def stream_summary(stream) -> list[dict[str, str | float | int]]:
    """Tidy per-trace summary of an ObsPy stream."""
    return [
        {
            "trace_id": str(tr.id),
            "start": format_timestamp(float(tr.stats.starttime.timestamp)),
            "end": format_timestamp(float(tr.stats.endtime.timestamp)),
            "sampling_rate": float(tr.stats.sampling_rate),
            "samples": int(tr.stats.npts),
        }
        for tr in stream
    ]


def repick_stream(
    stream,
    model_name: str,
    weights: str,
    min_confidence: float = 0.0,
    detrend: bool = True,
) -> list[CandidatePick]:
    """Run the official ``model.classify(stream)`` API on the CPU.

    Returns the model's real candidate picks with their confidence; the list is
    empty when the model proposed nothing above ``min_confidence``.  Nothing is
    applied to the session here — the user must select and stage explicitly.
    """
    model = load_picker(model_name, weights)
    try:
        work = stream.copy()
        if detrend:
            work.detrend("linear")
        outputs = model.classify(work)
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise SeisBenchError(
            "inference_failed",
            f"{model_name}/{weights} inference failed ({e}); check that the "
            "stream has the channels the model expects and enough samples",
        ) from e

    picks = getattr(outputs, "picks", outputs)
    candidates: list[CandidatePick] = []
    for pick in picks:
        confidence = float(getattr(pick, "peak_value", float("nan")) or 0.0)
        if math.isfinite(confidence) and confidence < min_confidence:
            continue
        trace_id = str(getattr(pick, "trace_id", ""))
        station_id = (
            trace_id.split(".")[1] if trace_id.count(".") >= 1 else trace_id
        )
        candidates.append(
            CandidatePick(
                trace_id=trace_id,
                station_id=station_id,
                phase=str(getattr(pick, "phase", "?")).upper(),
                time=float(getattr(pick, "peak_time").timestamp),
                confidence=confidence,
                model=model_name,
                weights=weights,
            )
        )
    candidates.sort(key=lambda c: (c.station_id, c.phase, -c.confidence))
    return candidates


def stage_candidate_pair(
    session: DashboardSession,
    event_id: str,
    station_id: str,
    p_candidate: CandidatePick | None,
    s_candidate: CandidatePick | None,
    note: str = "",
) -> PickEdit:
    """Stage an explicitly selected model P/S pair onto one station's pick."""
    if p_candidate is None and s_candidate is None:
        raise PickEditError(
            "select at least one candidate pick before applying — model "
            "predictions are never applied automatically"
        )
    provenance = (p_candidate or s_candidate).provenance
    detail = " · ".join(
        f"{c.phase} {c.confidence:.3f}"
        for c in (p_candidate, s_candidate)
        if c is not None
    )
    return session.stage_edit(
        event_id,
        station_id,
        p_time=format_timestamp(p_candidate.time) if p_candidate else None,
        s_time=format_timestamp(s_candidate.time) if s_candidate else None,
        provenance=provenance,
        note=note or f"selected candidate(s): {detail}",
    )


# ---------------------------------------------------------------------------
# the Wadati figure (the centrepiece)
# ---------------------------------------------------------------------------
def wadati_figure(
    session: DashboardSession,
    event_id: str,
    width: float = 7.2,
    height: float = 4.4,
):  # pragma: no cover - visual
    """Matplotlib Wadati figure: teal retained, red outliers, amber revised."""
    import matplotlib as mpl
    import numpy as np
    from matplotlib.figure import Figure

    mpl.rcParams.update(
        {
            "figure.facecolor": THEME["paper"],
            "axes.facecolor": THEME["paper"],
            "axes.edgecolor": THEME["rule"],
            "axes.labelcolor": THEME["ink"],
            "axes.titlecolor": THEME["ink"],
            "grid.color": THEME["rule"],
            "text.color": THEME["ink"],
            "xtick.color": THEME["slate"],
            "ytick.color": THEME["slate"],
            "legend.frameon": False,
        }
    )

    result = session.current(event_id)
    picks = session.picks(event_id)
    fig = Figure(figsize=(width, height), dpi=110)
    ax = fig.subplots()
    ax.grid(True, lw=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("P travel time  $t_P$  (s)")
    ax.set_ylabel("$t_S - t_P$  (s)")

    if not picks or result is None:
        ax.set_title("no picks for this event", loc="left", fontsize=11)
        return fig

    outliers = set(result.removed_stations)
    fit = result.fit if result.fit.valid else result.initial_fit
    xs = [p.p_travel_time for p in picks]

    if fit.valid:
        grid = np.linspace(0.0, max(xs) * 1.08, 50)
        ax.plot(
            grid,
            fit.intercept + fit.slope * grid,
            color=THEME["teal"] if result.accepted else THEME["red"],
            lw=1.4,
            zorder=2,
            label=f"fit  Vp/Vs = {fit.vp_vs:.3f}   r = {fit.pearson_r:.4f}",
        )

    groups = {
        "retained": (
            [
                p
                for p in picks
                if p.station_id not in outliers and not p.revised
            ],
            THEME["teal"],
            "o",
            44,
        ),
        "outlier": (
            [p for p in picks if p.station_id in outliers and not p.revised],
            THEME["red"],
            "X",
            56,
        ),
        "revised": ([p for p in picks if p.revised], THEME["amber"], "D", 52),
    }
    for label, (subset, colour, marker, size) in groups.items():
        if not subset:
            continue
        ax.scatter(
            [p.p_travel_time for p in subset],
            [p.s_minus_p for p in subset],
            s=size,
            marker=marker,
            zorder=3,
            facecolor=colour,
            edgecolor=THEME["paper"],
            lw=0.8,
            label=f"{label} (n={len(subset)})",
        )
    for pick in picks:
        colour = (
            THEME["amber"]
            if pick.revised
            else (
                THEME["red"] if pick.station_id in outliers else THEME["slate"]
            )
        )
        ax.annotate(
            pick.station_id,
            (pick.p_travel_time, pick.s_minus_p),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7.5,
            color=colour,
        )

    ax.set_title(
        f"Wadati diagram · {event_id} · {result.status} / {result.reason}",
        loc="left",
        fontsize=11.5,
        pad=10,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# the Panel workspace
# ---------------------------------------------------------------------------
def _card(*objects, tone: str = "paper"):  # pragma: no cover - visual
    import panel as pn

    return pn.Column(
        *objects,
        styles={
            "background": THEME[tone],
            "padding": "12px 14px",
            "border": f"1px solid {THEME['rule']}",
        },
    )


def _note(text: str, tone: str = "slate"):  # pragma: no cover - visual
    import panel as pn

    return pn.pane.Markdown(text, styles={"color": THEME[tone]})


def _rule(text: str):  # pragma: no cover - visual
    import panel as pn

    return pn.pane.HTML(
        f"<div style='border-top:1px solid {THEME['rule']};margin-top:6px;"
        f"padding-top:6px;font-size:10px;letter-spacing:.24em;"
        f"text-transform:uppercase;color:#8a7f68'>{text}</div>"
    )


def build_dashboard(
    session: DashboardSession,
    export_dir: str | Path | None = None,
):  # pragma: no cover - visual
    """Assemble the embedded Panel dashboard for ``session``."""
    import pandas as pd
    import panel as pn

    state: dict[str, object] = {"candidates": [], "stream": None}

    # --- navigation ------------------------------------------------------
    status_filter = pn.widgets.RadioButtonGroup(
        name="status filter",
        options=list(FILTERS),
        value="all",
        button_type="default",
    )
    event_select = pn.widgets.Select(
        name="Event", options=session.event_ids(), value=session.event_ids()[0]
    )
    prev_button = pn.widgets.Button(name="◀ previous", width=110)
    next_button = pn.widgets.Button(name="next ▶", width=110)

    plot_pane = pn.pane.Matplotlib(
        wadati_figure(session, event_select.value), tight=True, format="svg"
    )
    metrics_pane = pn.pane.Markdown("")
    report_pane = pn.pane.Markdown("")
    table_pane = pn.pane.DataFrame(pd.DataFrame(), index=False, width=980)
    edit_log_pane = pn.pane.DataFrame(pd.DataFrame(), index=False, width=980)
    compare_pane = pn.pane.DataFrame(pd.DataFrame(), index=False, width=560)
    dataset_pane = pn.pane.DataFrame(pd.DataFrame(), index=False, width=460)
    message_pane = _note("")

    station_select = pn.widgets.Select(name="Station", options=[])
    p_input = pn.widgets.TextInput(name="P travel time (s) or ISO P time")
    s_input = pn.widgets.TextInput(name="S−P (s) or ISO S time")
    note_input = pn.widgets.TextInput(
        name="Provenance note", placeholder="why?"
    )
    stage_button = pn.widgets.Button(
        name="Validate + stage edit", button_type="primary"
    )
    apply_button = pn.widgets.Button(name="Apply staged", button_type="success")
    discard_button = pn.widgets.Button(name="Discard staged")
    undo_button = pn.widgets.Button(name="Undo last applied")
    reset_button = pn.widgets.Button(name="Reset event")
    reset_all_button = pn.widgets.Button(name="Reset dataset")
    pending_pane = _note("no staged edits")

    def say(text: str, tone: str = "slate") -> None:
        message_pane.object = text
        message_pane.styles = {"color": THEME[tone]}

    def _numeric(text: str) -> tuple[float | None, str | None]:
        """Split a field into (numeric, iso) — an ISO string contains ``-``/``T``."""
        raw = (text or "").strip()
        if not raw:
            return None, None
        try:
            return float(raw), None
        except ValueError:
            return None, raw

    def refresh(_event=None) -> None:
        event_id = event_select.value
        result = session.current(event_id)
        picks = session.picks(event_id)
        plot_pane.object = wadati_figure(session, event_id)
        table_pane.object = pd.DataFrame(session.pick_table(event_id))
        station_select.options = session.station_ids(event_id)
        edit_log_pane.object = pd.DataFrame(session.edit_log_rows())
        compare_pane.object = pd.DataFrame(session.comparison_rows(event_id))
        dataset_pane.object = pd.DataFrame(session.dataset_comparison())
        staged = session.pending_edits(event_id)
        pending_pane.object = (
            "no staged edits"
            if not staged
            else "**staged:** " + " · ".join(e.describe() for e in staged)
        )
        pending_pane.styles = {"color": THEME["amber" if staged else "slate"]}
        if result is None:
            metrics_pane.object = "no QC result for this event"
            report_pane.object = ""
            return
        fit = result.fit if result.fit.valid else result.initial_fit
        tone = THEME["teal"] if result.accepted else THEME["red"]
        revised = sum(1 for p in picks if p.revised)
        metrics_pane.object = (
            f"### `{event_id}` — <span style='color:{tone}'>"
            f"{result.status.upper()}</span> · `{result.reason}`\n\n"
            f"`Vp/Vs = {fit.vp_vs:.3f} ± {fit.vp_vs_stderr:.3f}`  ·  "
            f"`r = {fit.pearson_r:.4f}`  ·  `r² = {fit.r_squared:.4f}`  ·  "
            f"`rmse = {fit.rmse:.3f}`  ·  retained "
            f"**{result.retained_stations}/{result.original_stations}**  ·  "
            f"depth **{result.search_depth}**  ·  "
            f"{result.combinations_evaluated:,} combination(s)\n\n"
            f"outliers: {', '.join(result.outlier_stations) or 'none'}  ·  "
            f"revised picks: **{revised}**  ·  "
            f"{'second pass' if event_id in session.revised_events else 'first pass'}"
        )
        report_pane.object = "```\n" + result.report() + "\n```"

    def on_filter(_event=None) -> None:
        ids = session.filtered_event_ids(status_filter.value)
        if not ids:
            say(f"no events match the `{status_filter.value}` filter", "red")
            return
        event_select.options = ids
        if event_select.value not in ids:
            event_select.value = ids[0]
        else:
            refresh()

    def step(delta: int):
        def handler(_event=None) -> None:
            event_select.value = session.neighbour(
                event_select.value, delta, status_filter.value
            )

        return handler

    def on_stage(_event=None) -> None:
        p_number, p_iso = _numeric(p_input.value)
        s_number, s_iso = _numeric(s_input.value)
        try:
            edit = session.stage_edit(
                event_select.value,
                station_select.value,
                p_travel_time=p_number,
                s_minus_p=s_number,
                p_time=p_iso,
                s_time=s_iso,
                provenance="manual",
                note=note_input.value,
            )
        except PickEditError as exc:
            logging.exception("Unexpected error")
            say(f"**rejected** — {exc}", "red")
            return
        say(f"staged — {edit.describe()}", "amber")
        refresh()

    def on_apply(_event=None) -> None:
        applied = session.apply_pending(event_select.value)
        if not applied:
            say("nothing staged to apply", "red")
            return
        say(
            f"applied {len(applied)} edit(s) — rerun to see the second pass",
            "amber",
        )
        refresh()

    def on_discard(_event=None) -> None:
        dropped = session.discard_pending(event_select.value)
        say(f"discarded {len(dropped)} staged edit(s)")
        refresh()

    def on_undo(_event=None) -> None:
        edit = session.undo_last(event_select.value)
        say(
            f"undone — {edit.describe()}"
            if edit
            else "no applied edit to undo",
            "amber" if edit else "red",
        )
        refresh()

    def on_reset(_event=None) -> None:
        count = session.reset_event(event_select.value)
        say(f"reset {count} revised pick(s) for {event_select.value}")
        refresh()

    def on_reset_all(_event=None) -> None:
        count = session.reset_all()
        say(f"reset the whole dataset ({count} revised pick(s))")
        refresh()

    status_filter.param.watch(lambda *_: on_filter(), "value")
    event_select.param.watch(lambda *_: refresh(), "value")
    prev_button.on_click(step(-1))
    next_button.on_click(step(1))
    stage_button.on_click(on_stage)
    apply_button.on_click(on_apply)
    discard_button.on_click(on_discard)
    undo_button.on_click(on_undo)
    reset_button.on_click(on_reset)
    reset_all_button.on_click(on_reset_all)

    # --- second pass -----------------------------------------------------
    rerun_event_button = pn.widgets.Button(
        name="Rerun this event (second pass)", button_type="primary"
    )
    rerun_all_button = pn.widgets.Button(
        name="Rerun the full dataset", button_type="primary"
    )

    def on_rerun_event(_event=None) -> None:
        rerun_event_button.loading = True
        try:
            result = session.rerun_event(event_select.value)
            say(
                f"second pass: {result.status} / {result.reason} · "
                f"Vp/Vs {result.fit.vp_vs:.3f}",
                "teal" if result.accepted else "red",
            )
            refresh()
        finally:
            rerun_event_button.loading = False

    def on_rerun_all(_event=None) -> None:
        rerun_all_button.loading = True
        try:
            dataset = session.rerun_all()
            summary = dataset.summary()
            say(
                f"second pass over {summary['events']} event(s): "
                f"{summary['accepted']} accepted / {summary['rejected']} rejected",
                "teal",
            )
            refresh()
        finally:
            rerun_all_button.loading = False

    rerun_event_button.on_click(on_rerun_event)
    rerun_all_button.on_click(on_rerun_all)

    downloads = pn.Column(
        pn.Row(
            pn.widgets.FileDownload(
                callback=lambda: io.StringIO(
                    session.export_csv("original", "events")
                ),
                filename="wadati_original_events_qc.csv",
                label="Original event QC (CSV)",
            ),
            pn.widgets.FileDownload(
                callback=lambda: io.StringIO(
                    session.export_csv("original", "picks")
                ),
                filename="wadati_original_picks_qc.csv",
                label="Original pick QC (CSV)",
            ),
        ),
        pn.Row(
            pn.widgets.FileDownload(
                callback=lambda: io.StringIO(
                    session.export_csv("revised", "events")
                ),
                filename="wadati_revised_events_qc.csv",
                label="Revised event QC (CSV)",
                button_type="primary",
            ),
            pn.widgets.FileDownload(
                callback=lambda: io.StringIO(
                    session.export_csv("revised", "picks")
                ),
                filename="wadati_revised_picks_qc.csv",
                label="Revised pick QC (CSV)",
                button_type="primary",
            ),
        ),
    )
    if export_dir is not None:
        written = session.write_exports(export_dir)
        downloads.append(
            _note(
                "on disk: "
                + " · ".join(f"`{p.name}`" for p in written.values())
            )
        )

    # --- SeisBench assist ------------------------------------------------
    model_select = pn.widgets.Select(
        name="Model class",
        options=list(SUPPORTED_MODELS),
        value=SUPPORTED_MODELS[0],
    )
    weight_select = pn.widgets.Select(name="Pretrained weights", options=[])
    weights_button = pn.widgets.Button(name="List pretrained weights")
    mseed_upload = pn.widgets.FileInput(accept=".mseed,.msd,.seed")
    mseed_path = pn.widgets.TextInput(
        name="…or a local MiniSEED path", placeholder="data/LO.LOBH.mseed"
    )
    confidence_slider = pn.widgets.FloatSlider(
        name="Minimum confidence", start=0.0, end=0.95, step=0.05, value=0.3
    )
    run_button = pn.widgets.Button(
        name="Run CPU inference", button_type="primary"
    )
    candidate_pane = pn.pane.DataFrame(pd.DataFrame(), index=False, width=680)
    p_candidate_select = pn.widgets.Select(
        name="Apply as P", options=["— none —"]
    )
    s_candidate_select = pn.widgets.Select(
        name="Apply as S", options=["— none —"]
    )
    stage_candidate_button = pn.widgets.Button(
        name="Stage selected candidate(s)", button_type="warning"
    )
    sb_message = _note(
        "Nothing is loaded until a button is pressed. Weights download to "
        "`~/.seisbench/models` on first use."
    )

    def sb_say(text: str, tone: str = "slate") -> None:
        sb_message.object = text
        sb_message.styles = {"color": THEME[tone]}

    def on_weights(_event=None) -> None:
        weights_button.loading = True
        try:
            names = available_weights(model_select.value)
            weight_select.options = names
            if names:
                weight_select.value = names[0]
            sb_say(
                f"{len(names)} pretrained weight set(s) for {model_select.value}",
                "teal",
            )
        except SeisBenchError as exc:
            logging.exception("Unexpected error")
            weight_select.options = []
            sb_say(f"**{exc.code}** — {exc.message}", "red")
        finally:
            weights_button.loading = False

    def _candidate_label(candidate: CandidatePick) -> str:
        return (
            f"{candidate.station_id} · {candidate.phase} · "
            f"{format_timestamp(candidate.time)} · p={candidate.confidence:.3f}"
        )

    def on_run(_event=None) -> None:
        run_button.loading = True
        try:
            if mseed_upload.value:
                stream = read_mseed(mseed_upload.value)
            elif mseed_path.value.strip():
                stream = read_mseed(mseed_path.value.strip())
            else:
                sb_say(
                    "upload a MiniSEED file or give a local path first — there are "
                    "no synthetic waveforms here",
                    "red",
                )
                return
            state["stream"] = stream
            candidates = repick_stream(
                stream,
                model_select.value,
                weight_select.value,
                min_confidence=confidence_slider.value,
            )
            state["candidates"] = candidates
            candidate_pane.object = pd.DataFrame(
                [c.as_row() for c in candidates]
            )
            labels = ["— none —"] + [_candidate_label(c) for c in candidates]
            p_candidate_select.options = labels
            s_candidate_select.options = labels
            p_candidate_select.value = "— none —"
            s_candidate_select.value = "— none —"
            if candidates:
                sb_say(
                    f"{len(candidates)} candidate pick(s) from "
                    f"{model_select.value}/{weight_select.value} over "
                    f"{len(stream)} trace(s) — select and stage explicitly; "
                    "nothing was written to any pick",
                    "teal",
                )
            else:
                sb_say(
                    f"{model_select.value}/{weight_select.value} proposed no pick "
                    f"above p ≥ {confidence_slider.value:.2f} — lower the threshold "
                    "or check the waveform window",
                    "amber",
                )
        except SeisBenchError as exc:
            logging.exception("Unexpected error")
            candidate_pane.object = pd.DataFrame()
            sb_say(f"**{exc.code}** — {exc.message}", "red")
        finally:
            run_button.loading = False

    def _selected(select) -> CandidatePick | None:
        candidates: list[CandidatePick] = state["candidates"]  # type: ignore[assignment]
        for candidate in candidates:
            if _candidate_label(candidate) == select.value:
                return candidate
        return None

    def on_stage_candidate(_event=None) -> None:
        p_candidate = _selected(p_candidate_select)
        s_candidate = _selected(s_candidate_select)
        target = p_candidate or s_candidate
        station = station_select.value
        if target is not None and target.station_id in session.station_ids(
            event_select.value
        ):
            station = target.station_id
        try:
            edit = stage_candidate_pair(
                session,
                event_select.value,
                station,
                p_candidate,
                s_candidate,
            )
        except PickEditError as exc:
            logging.exception("Unexpected error")
            sb_say(f"**rejected** — {exc}", "red")
            return
        sb_say(f"staged — {edit.describe()}", "amber")
        refresh()

    weights_button.on_click(on_weights)
    run_button.on_click(on_run)
    stage_candidate_button.on_click(on_stage_candidate)

    # --- discovery (real FDSN, button driven) ----------------------------
    discovery_card = _discovery_card()

    # --- layout ----------------------------------------------------------
    left = pn.Column(
        _card(
            pn.pane.Markdown(
                "### Wadati workspace",
                styles={"color": THEME["ink"]},
            ),
            pn.Row(status_filter, prev_button, next_button),
            event_select,
            plot_pane,
            metrics_pane,
            _note(
                f"<span style='color:{THEME['teal']}'>■ retained</span> · "
                f"<span style='color:{THEME['red']}'>✕ outlier</span> · "
                f"<span style='color:{THEME['amber']}'>◆ revised pick</span>"
            ),
        ),
        _card(
            _rule("editable pick table"),
            table_pane,
            pn.Row(station_select, p_input, s_input),
            pn.Row(note_input, stage_button),
            pn.Row(
                apply_button,
                discard_button,
                undo_button,
                reset_button,
                reset_all_button,
            ),
            pending_pane,
            message_pane,
            tone="paper_deep",
        ),
        _card(
            _rule("second pass · original vs revised"),
            pn.Row(rerun_event_button, rerun_all_button),
            pn.Row(compare_pane, dataset_pane),
            downloads,
        ),
        _card(
            _rule("applied edit log · provenance"),
            edit_log_pane,
            tone="paper_deep",
        ),
        _card(_rule("per-event report"), report_pane),
    )

    right = pn.Column(
        _card(
            pn.pane.Markdown(
                "### Waveform-assisted re-picking (optional)",
                styles={"color": THEME["ink"]},
            ),
            _note(
                "Real SeisBench models on the **CPU**, through the official "
                "`model.classify(stream)` API. The model is loaded only when you "
                "press a button, and a candidate pick is applied only when you "
                "select it and stage it — picks are never overwritten silently."
            ),
            pn.Row(model_select, weights_button),
            weight_select,
            mseed_upload,
            mseed_path,
            confidence_slider,
            run_button,
            candidate_pane,
            pn.Row(p_candidate_select, s_candidate_select),
            stage_candidate_button,
            sb_message,
            tone="paper_deep",
        ),
        discovery_card,
        width=560,
    )

    refresh()
    return pn.Column(
        pn.pane.Markdown(
            "## Wadati QC dashboard — event-by-event inspection, re-picking and "
            "second pass",
            styles={"color": THEME["ink"]},
        ),
        _note(session.config.describe()),
        pn.Row(left, right),
        styles={"background": THEME["paper"], "padding": "10px"},
    )


def _discovery_card():  # pragma: no cover - visual
    """Real LO / OSPL station and Hispaniola event selection (button driven)."""
    import pandas as pd
    import panel as pn

    try:
        from app.notebook.discovery import (
            DEFAULT_MIN_MAGNITUDE,
            DEFAULT_WINDOW_DAYS,
            EVENT_PROVIDER,
            HISPANIOLA_BBOX,
            LO_NETWORK,
            STATION_PROVIDER,
            default_window,
            fetch_hispaniola_events,
            fetch_lo_stations,
        )
    except ImportError:
        from wadati_discovery import (  # type: ignore[no-redef]
            DEFAULT_MIN_MAGNITUDE,
            DEFAULT_WINDOW_DAYS,
            EVENT_PROVIDER,
            HISPANIOLA_BBOX,
            LO_NETWORK,
            STATION_PROVIDER,
            default_window,
            fetch_hispaniola_events,
            fetch_lo_stations,
        )

    start, end = default_window(DEFAULT_WINDOW_DAYS)
    window = pn.widgets.DatetimeRangeInput(
        name="UTC window", value=(start.datetime, end.datetime)
    )
    magnitude = pn.widgets.FloatSlider(
        name="Minimum magnitude",
        start=1.0,
        end=7.0,
        step=0.1,
        value=DEFAULT_MIN_MAGNITUDE,
    )
    station_button = pn.widgets.Button(
        name=f"Query {LO_NETWORK} stations ({STATION_PROVIDER})"
    )
    event_button = pn.widgets.Button(
        name=f"Query Hispaniola events ({EVENT_PROVIDER})"
    )
    station_picker = pn.widgets.MultiChoice(
        name="LO / OSPL stations (Dominican Republic)", options=[], value=[]
    )
    event_picker = pn.widgets.MultiSelect(
        name="Hispaniola-region earthquakes", options=[], value=[], size=8
    )
    station_table = pn.pane.DataFrame(pd.DataFrame(), index=False, width=520)
    event_table = pn.pane.DataFrame(pd.DataFrame(), index=False, width=520)
    message = _note("Nothing is queried until a button is pressed.")

    def say(text: str, tone: str = "slate") -> None:
        message.object = text
        message.styles = {"color": THEME[tone]}

    def on_stations(_event=None) -> None:
        station_button.loading = True
        try:
            begin, finish = window.value
            result = fetch_lo_stations(
                starttime=begin, endtime=finish, bbox=HISPANIOLA_BBOX
            )
            station_table.object = (
                result.to_dataframe() if result.rows else pd.DataFrame()
            )
            options = result.options()
            station_picker.options = options
            station_picker.value = list(options.values())[:6]
            say(
                "```\n" + result.report() + "\n```",
                "teal" if result.rows else "red",
            )
        finally:
            station_button.loading = False

    def on_events(_event=None) -> None:
        event_button.loading = True
        try:
            begin, finish = window.value
            result = fetch_hispaniola_events(
                starttime=begin,
                endtime=finish,
                minmagnitude=magnitude.value,
                bbox=HISPANIOLA_BBOX,
            )
            event_table.object = (
                result.to_dataframe() if result.rows else pd.DataFrame()
            )
            options = result.options()
            event_picker.options = options
            event_picker.value = list(options.values())[:3]
            say(
                "```\n" + result.report() + "\n```",
                "teal" if result.rows else "red",
            )
        finally:
            event_button.loading = False

    station_button.on_click(on_stations)
    event_button.on_click(on_events)

    return _card(
        pn.pane.Markdown(
            "### Real data selection — LO / OSPL and Hispaniola",
            styles={"color": THEME["ink"]},
        ),
        _note(
            "**Hispaniola** is the Caribbean island shared by the **Dominican "
            "Republic** and **Haiti**. Network **LO** is the *Observatorio "
            "Sismológico Politécnico Loyola* (**OSPL**) network of the "
            "**Dominican Republic**, served by **EARTHSCOPE** only; regional "
            "earthquakes come from **USGS**."
        ),
        window,
        magnitude,
        pn.Row(station_button, event_button),
        station_picker,
        station_table,
        event_picker,
        event_table,
        message,
    )


# ---------------------------------------------------------------------------
# offline self-check
# ---------------------------------------------------------------------------
def self_check() -> dict[str, str | int | float]:
    """Exercise every logic path offline; returns a tidy outcome summary."""
    try:  # sample data, in-app or notebook-local
        from app.notebook.ingest import load_csv_text
        from app.notebook.samples import sample_long_csv
    except ImportError:  # pragma: no cover - notebook copy
        from wadati_ingest import load_csv_text  # type: ignore
        from wadati_samples import sample_long_csv  # type: ignore

    ingested = load_csv_text(sample_long_csv())
    session = DashboardSession(ingested, SearchConfig())
    assert session.event_ids(), "no events reached the dashboard session"
    event_id = session.event_ids()[0]
    station = session.station_ids(event_id)[0]

    # filters
    assert set(session.filtered_event_ids("all")) == set(session.event_ids())
    assert not session.filtered_event_ids("edited")
    assert (
        session.neighbour(event_id, 1) != event_id
        or len(session.event_ids()) == 1
    )

    # validation
    for kwargs, fragment in (
        ({"p_travel_time": -1.0}, "positive"),
        ({"s_minus_p": 0.0}, "positive"),
        ({"p_travel_time": float("nan")}, "finite"),
        ({}, "changes nothing"),
    ):
        try:
            session.stage_edit(event_id, station, **kwargs)
        except PickEditError as exc:
            logging.exception("Unexpected error")
            assert fragment in str(exc), (fragment, str(exc))
        else:  # pragma: no cover - guard
            raise AssertionError(f"{kwargs} should not validate")
    try:
        session.stage_edit(event_id, "NOT-A-STATION", p_travel_time=3.0)
    except PickEditError:
        logging.exception("Unexpected error")
    else:  # pragma: no cover - guard
        raise AssertionError("unknown station should not validate")

    # stage / apply / undo / reset with provenance
    baseline = session.pick(event_id, station).s_minus_p
    edit = session.stage_edit(
        event_id, station, s_minus_p=baseline + 1.25, note="late S re-read"
    )
    assert session.pending_edits(event_id) == [edit]
    assert not session.pick(event_id, station).revised
    assert session.apply_pending(event_id) == [edit]
    assert session.pick(event_id, station).revised
    assert session.pick(event_id, station).provenance == "manual"
    assert event_id in session.edited_event_ids()

    revised = session.rerun_event(event_id)
    assert revised.event_id == event_id
    assert session.has_revision
    assert session.comparison_rows(event_id)
    assert any(
        row["revised"] != "— not rerun —"
        for row in session.comparison_rows(event_id)
    )

    assert session.undo_last(event_id) == edit
    assert not session.pick(event_id, station).revised
    assert session.pick(event_id, station).provenance == "ingested"

    # absolute-time editing goes through the same validation
    pick = session.pick(event_id, station)
    assert pick.origin_time is not None, "sample CSV carries origin times"
    session.stage_edit(
        event_id,
        station,
        s_time=format_timestamp(pick.s_time + 0.4),
        note="absolute S re-read",
    )
    session.apply_pending(event_id)
    assert session.pick(event_id, station).revised
    assert session.reset_event(event_id) >= 1
    assert not session.pick(event_id, station).revised
    assert event_id not in session.revised_events

    # dataset-wide second pass and exports
    dataset = session.rerun_all()
    assert len(dataset.results) == len(session.event_ids())
    assert "event_id" in session.export_csv("original", "events")
    assert "state" in session.export_csv("revised", "picks")
    assert len(session.dataset_comparison()) == 9
    assert session.reset_all() == 0

    # SeisBench errors are truthful, and nothing is loaded implicitly
    for bad in ("NotAModel", "PhaseNetXL"):
        try:
            available_weights(bad)
        except SeisBenchError as exc:
            logging.exception("Unexpected error")
            assert exc.code in {"unsupported_model", "seisbench_missing"}
        else:  # pragma: no cover - guard
            raise AssertionError(f"{bad} should not be supported")
    try:
        read_mseed("does/not/exist.mseed")
    except SeisBenchError as exc:
        logging.exception("Unexpected error")
        assert exc.code == "read_failed"
    else:  # pragma: no cover - guard
        raise AssertionError("a missing waveform file must be reported")
    try:
        read_mseed(b"not a miniseed file")
    except SeisBenchError as exc:
        logging.exception("Unexpected error")
        assert exc.code == "read_failed"
    else:  # pragma: no cover - guard
        raise AssertionError("invalid MiniSEED bytes must be reported")
    assert not _MODEL_CACHE, "no model may be loaded by the self-check"

    return {
        "events": len(session.event_ids()),
        "accepted": len(session.original.accepted),
        "rejected": len(session.original.rejected),
        "reruns": len(session.revised_events),
        "applied_edits": len(session.applied),
        "source": session.source,
    }


def main() -> None:
    print(DASHBOARD_DOC)
    summary = self_check()
    print("dashboard self-check:", summary)
    print("Step 4 logic behaved exactly as documented (fully offline).")


if __name__ == "__main__":
    main()

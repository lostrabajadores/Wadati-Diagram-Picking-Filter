"""Validated ingestion and normalization of Wadati phase-pick data.

Pure-Python (stdlib only) apart from the optional QuakeML reader, which needs
ObsPy, and ``IngestResult.to_dataframe`` which needs pandas.  Everything here
is deliberately importable and testable outside the notebook: the notebook
generator copies this module next to the generated notebook so the notebook
exercises exactly this code.

Canonical schema
----------------
One normalized row per **event / station P–S pair**:

==================  ========  ==========================================
column              required  meaning
==================  ========  ==========================================
event_id            yes       catalog / origin identifier
station_id          yes       station code (SEED station, e.g. ``BRG``)
network             no        SEED network code (e.g. ``GR``)
channel             no        SEED channel code of the P pick
source              yes       provenance, e.g. ``csv:picks.csv``
origin_time         no        event origin time, ISO-8601 UTC
p_time              no        absolute P pick time, ISO-8601 UTC
s_time              no        absolute S pick time, ISO-8601 UTC
p_travel_time       yes       origin-relative P travel time, seconds
s_minus_p           yes       S−P interval, seconds (> 0)
==================  ========  ==========================================

``p_travel_time`` and ``s_minus_p`` are the two quantities the Wadati diagram
consumes.  They may be supplied directly as numeric travel times, or derived
from absolute timestamps plus an origin time.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_COLUMNS: list[str] = [
    "event_id",
    "station_id",
    "network",
    "channel",
    "source",
    "origin_time",
    "p_time",
    "s_time",
    "p_travel_time",
    "s_minus_p",
]

#: Minimum number of usable P–S pairs for an event to be worth a Wadati fit.
MIN_PAIRS_PER_EVENT: int = 4

P_PHASE_ALIASES = {"P", "PG", "PN", "PB", "P1", "PP"}
S_PHASE_ALIASES = {"S", "SG", "SN", "SB", "S1", "SS"}

SCHEMA_DOC = """\
CANONICAL PICK SCHEMA  (one row per event / station P-S pair)
------------------------------------------------------------
  event_id       str    required   catalog / origin identifier
  station_id     str    required   station code, e.g. BRG
  network        str    optional   SEED network code, e.g. GR
  channel        str    optional   SEED channel of the P pick, e.g. HHZ
  source         str    required   provenance, e.g. csv:picks.csv
  origin_time    str    optional   event origin time, ISO-8601 UTC
  p_time         str    optional   absolute P pick time, ISO-8601 UTC
  s_time         str    optional   absolute S pick time, ISO-8601 UTC
  p_travel_time  float  required   origin-relative P travel time (s)
  s_minus_p      float  required   S - P interval (s), must be > 0

ACCEPTED CSV LAYOUTS
--------------------
  long   event_id, station_id, phase, time            [+ origin_time]
         event_id, station_id, phase, travel_time
  wide   event_id, station_id, p_time, s_time         [+ origin_time]
         event_id, station_id, p_travel_time, s_minus_p
         event_id, station_id, p_travel_time, s_travel_time
  Optional in every layout: network, channel.

ACCEPTED QUAKEML
----------------
  Any file readable by obspy.read_events(path, format="quakeml").
  P/S picks are paired per event + station via pick.waveform_id; the phase
  label comes from pick.phase_hint, falling back to the matching
  origin arrival phase.  P travel time is measured against the preferred
  origin time, or the first available origin that carries a time.

RESULT API  (what load_csv_picks / load_quakeml_picks / load_picks return)
-------------------------------------------------------------------------
  IngestResult.pairs            list[PickPair]  every normalized pair
  IngestResult.usable_pairs     list[PickPair]  pairs of events clearing min_pairs
  IngestResult.event_ids        list[str]       events in first-seen order
  IngestResult.event_pairs(id)  list[PickPair]  pairs of one event
  IngestResult.event_summary()  list[dict]      per-event table (pairs, span,
                                                median Vp/Vs, usable flag)
  IngestResult.station_summary()list[dict]      per-station table
  IngestResult.as_rows()        list[dict]      canonical rows
  IngestResult.to_dataframe()   pandas.DataFrame of the canonical rows
  IngestResult.issues / errors / warnings       IngestIssue objects
  IngestResult.report()         str             printable ingestion report
  write_canonical_csv(result, path) -> Path     canonical table on disk

VALIDATION CODES
----------------
  malformed_time       timestamp or number could not be parsed
  unknown_phase        phase label is neither a P nor an S alias
  duplicate_phase      more than one P (or S) pick for one event + station
  missing_p            S pick present with no matching P pick
  missing_s            P pick present with no matching S pick
  s_not_after_p        S pick is at or before the P pick
  missing_origin_time  absolute picks with no origin time -> no travel time
  origin_time_conflict rows disagree about one event's origin time
  insufficient_pairs   event has fewer than min_pairs usable pairs
"""


def parse_timestamp(value: object) -> float:
    """Parse an ISO-8601 timestamp (``Z`` or offset, naive == UTC) to epoch s."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        raise ValueError("empty timestamp")
    normalized = text
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    normalized = (
        normalized.replace(" ", "T", 1)
        if " " in normalized[:11]
        else normalized
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{text!r} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def format_timestamp(epoch: float | None) -> str:
    """Render epoch seconds as an ISO-8601 UTC string (``""`` when absent)."""
    if epoch is None:
        return ""
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_number(value: object) -> float:
    """Parse a numeric field, rejecting blanks and non-finite values."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        raise ValueError("empty number")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{text!r} is not a number") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{text!r} is not finite")
    return number


def normalize_phase(label: object) -> str | None:
    """Map a phase label onto ``"P"`` / ``"S"``, or ``None`` when unknown."""
    text = str(label or "").strip().upper()
    if not text:
        return None
    if text in P_PHASE_ALIASES:
        return "P"
    if text in S_PHASE_ALIASES:
        return "S"
    if text[0] == "P":
        return "P"
    if text[0] == "S":
        return "S"
    return None


@dataclass(frozen=True)
class IngestIssue:
    """One actionable validation message."""

    code: str
    level: str  # "error" | "warning"
    message: str
    event_id: str = ""
    station_id: str = ""

    def __str__(self) -> str:
        where = " · ".join(p for p in (self.event_id, self.station_id) if p)
        prefix = "ERROR  " if self.level == "error" else "warning"
        return f"[{prefix}] {self.code}: {self.message}" + (
            f"  ({where})" if where else ""
        )


@dataclass
class PickPair:
    """A normalized P/S pair for one event and station."""

    event_id: str
    station_id: str
    network: str
    channel: str
    source: str
    origin_time: float | None
    p_time: float | None
    s_time: float | None
    p_travel_time: float
    s_minus_p: float

    @property
    def apparent_vp_vs(self) -> float:
        """Single-pair Vp/Vs implied by this pair (diagnostic only)."""
        if self.p_travel_time <= 0.0:
            return float("nan")
        return 1.0 + self.s_minus_p / self.p_travel_time

    def as_row(self) -> dict[str, str | float]:
        return {
            "event_id": self.event_id,
            "station_id": self.station_id,
            "network": self.network,
            "channel": self.channel,
            "source": self.source,
            "origin_time": format_timestamp(self.origin_time),
            "p_time": format_timestamp(self.p_time),
            "s_time": format_timestamp(self.s_time),
            "p_travel_time": round(self.p_travel_time, 4),
            "s_minus_p": round(self.s_minus_p, 4),
        }


@dataclass
class IngestResult:
    """Normalized pairs plus every validation message raised while reading."""

    source: str
    pairs: list[PickPair] = field(default_factory=list)
    issues: list[IngestIssue] = field(default_factory=list)
    unusable_events: set[str] = field(default_factory=set)

    # --- issue views -----------------------------------------------------
    @property
    def errors(self) -> list[IngestIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[IngestIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return bool(self.usable_pairs)

    # --- pair views ------------------------------------------------------
    @property
    def usable_pairs(self) -> list[PickPair]:
        return [p for p in self.pairs if p.event_id not in self.unusable_events]

    @property
    def event_ids(self) -> list[str]:
        seen: list[str] = []
        for pair in self.pairs:
            if pair.event_id not in seen:
                seen.append(pair.event_id)
        return seen

    def event_pairs(self, event_id: str) -> list[PickPair]:
        return [p for p in self.pairs if p.event_id == event_id]

    def as_rows(self) -> list[dict[str, str | float]]:
        return [p.as_row() for p in self.pairs]

    # --- summaries -------------------------------------------------------
    def event_summary(self) -> list[dict[str, str | int | float | bool]]:
        rows: list[dict[str, str | int | float | bool]] = []
        for event_id in self.event_ids:
            pairs = self.event_pairs(event_id)
            errors = [i for i in self.errors if i.event_id == event_id]
            warns = [i for i in self.warnings if i.event_id == event_id]
            ratios = [p.apparent_vp_vs for p in pairs]
            rows.append(
                {
                    "event_id": event_id,
                    "source": pairs[0].source,
                    "origin_time": format_timestamp(pairs[0].origin_time),
                    "stations": len({p.station_id for p in pairs}),
                    "pairs": len(pairs),
                    "p_travel_min": round(
                        min(p.p_travel_time for p in pairs), 3
                    ),
                    "p_travel_max": round(
                        max(p.p_travel_time for p in pairs), 3
                    ),
                    "vp_vs_median": round(sorted(ratios)[len(ratios) // 2], 3),
                    "errors": len(errors),
                    "warnings": len(warns),
                    "usable": event_id not in self.unusable_events,
                }
            )
        return rows

    def station_summary(self) -> list[dict[str, str | int | float]]:
        by_station: dict[str, list[PickPair]] = {}
        for pair in self.pairs:
            by_station.setdefault(pair.station_id, []).append(pair)
        rows: list[dict[str, str | int | float]] = []
        for station_id in sorted(by_station):
            pairs = by_station[station_id]
            ratios = [p.apparent_vp_vs for p in pairs]
            rows.append(
                {
                    "station_id": station_id,
                    "network": pairs[0].network,
                    "channel": pairs[0].channel,
                    "events": len({p.event_id for p in pairs}),
                    "pairs": len(pairs),
                    "s_minus_p_mean": round(
                        sum(p.s_minus_p for p in pairs) / len(pairs), 3
                    ),
                    "vp_vs_mean": round(sum(ratios) / len(ratios), 3),
                }
            )
        return rows

    def report(self) -> str:
        """Human-readable ingestion report, safe to ``print`` in a notebook."""
        lines = [
            f"source            {self.source}",
            f"normalized pairs  {len(self.pairs)}  "
            f"(usable {len(self.usable_pairs)} across "
            f"{len(set(self.event_ids) - self.unusable_events)} event(s))",
            f"errors            {len(self.errors)}",
            f"warnings          {len(self.warnings)}",
        ]
        if self.issues:
            lines.append("")
            lines.extend(str(issue) for issue in self.issues)
        return "\n".join(lines)

    def to_dataframe(self):  # pragma: no cover - convenience for notebooks
        import pandas as pd

        frame = pd.DataFrame(self.as_rows(), columns=CANONICAL_COLUMNS)
        return frame


# ---------------------------------------------------------------------------
# internal record form used by every loader before normalization
# ---------------------------------------------------------------------------
@dataclass
class _Record:
    event_id: str
    station_id: str
    phase: str
    time: float | None = None
    travel_time: float | None = None
    origin_time: float | None = None
    network: str = ""
    channel: str = ""
    row: int = 0


def _normalize(
    records: list[_Record],
    source: str,
    issues: list[IngestIssue],
    min_pairs: int,
) -> IngestResult:
    """Pair P/S records per event + station and validate them."""
    result = IngestResult(source=source, issues=issues)

    # --- one origin time per event ---------------------------------------
    origins: dict[str, float | None] = {}
    for rec in records:
        if rec.origin_time is None:
            origins.setdefault(rec.event_id, None)
            continue
        known = origins.get(rec.event_id)
        if known is None:
            origins[rec.event_id] = rec.origin_time
        elif abs(known - rec.origin_time) > 1e-3:
            issues.append(
                IngestIssue(
                    "origin_time_conflict",
                    "warning",
                    f"rows disagree about the origin time ({format_timestamp(known)} vs "
                    f"{format_timestamp(rec.origin_time)}); keeping the first value",
                    rec.event_id,
                )
            )

    grouped: dict[tuple[str, str], dict[str, list[_Record]]] = {}
    order: list[tuple[str, str]] = []
    for rec in records:
        key = (rec.event_id, rec.station_id)
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        grouped[key].setdefault(rec.phase, []).append(rec)

    for event_id, station_id in order:
        phases = grouped[(event_id, station_id)]
        origin_time = origins.get(event_id)

        for phase in ("P", "S"):
            if len(phases.get(phase, [])) > 1:
                rows = ", ".join(str(r.row) for r in phases[phase])
                issues.append(
                    IngestIssue(
                        "duplicate_phase",
                        "error",
                        f"{len(phases[phase])} {phase} picks for one event/station "
                        f"(rows {rows}); keeping the earliest — de-duplicate the input",
                        event_id,
                        station_id,
                    )
                )
                phases[phase] = [
                    sorted(
                        phases[phase],
                        key=lambda r: (
                            r.time
                            if r.time is not None
                            else r.travel_time or 0.0
                        ),
                    )[0]
                ]

        p_rec = phases.get("P", [None])[0]
        s_rec = phases.get("S", [None])[0]
        if p_rec is None:
            issues.append(
                IngestIssue(
                    "missing_p",
                    "error",
                    "S pick has no matching P pick for this station; "
                    "add the P pick or drop the S pick",
                    event_id,
                    station_id,
                )
            )
            continue
        if s_rec is None:
            issues.append(
                IngestIssue(
                    "missing_s",
                    "error",
                    "P pick has no matching S pick for this station; "
                    "a Wadati point needs both phases",
                    event_id,
                    station_id,
                )
            )
            continue

        # --- travel times -------------------------------------------------
        if p_rec.travel_time is not None and s_rec.travel_time is not None:
            p_travel = p_rec.travel_time
            s_minus_p = s_rec.travel_time - p_rec.travel_time
        elif p_rec.time is not None and s_rec.time is not None:
            s_minus_p = s_rec.time - p_rec.time
            if origin_time is None:
                issues.append(
                    IngestIssue(
                        "missing_origin_time",
                        "error",
                        "absolute P/S picks but no origin time for this event, so the "
                        "origin-relative P travel time cannot be derived; supply "
                        "origin_time or numeric travel times",
                        event_id,
                        station_id,
                    )
                )
                continue
            p_travel = p_rec.time - origin_time
        else:
            issues.append(
                IngestIssue(
                    "malformed_time",
                    "error",
                    "P and S must both be absolute timestamps or both numeric travel "
                    "times; this pair mixes the two",
                    event_id,
                    station_id,
                )
            )
            continue

        if s_minus_p <= 0.0:
            issues.append(
                IngestIssue(
                    "s_not_after_p",
                    "error",
                    f"S is not later than P (S−P = {s_minus_p:.3f} s); the phases are "
                    "most likely swapped",
                    event_id,
                    station_id,
                )
            )
            continue
        if p_travel <= 0.0:
            issues.append(
                IngestIssue(
                    "malformed_time",
                    "error",
                    f"P travel time is not positive ({p_travel:.3f} s); check the "
                    "origin time against the pick time",
                    event_id,
                    station_id,
                )
            )
            continue

        result.pairs.append(
            PickPair(
                event_id=event_id,
                station_id=station_id,
                network=p_rec.network or s_rec.network,
                channel=p_rec.channel or s_rec.channel,
                source=source,
                origin_time=origin_time,
                p_time=p_rec.time,
                s_time=s_rec.time,
                p_travel_time=p_travel,
                s_minus_p=s_minus_p,
            )
        )

    result.pairs.sort(key=lambda p: (p.event_id, p.p_travel_time))

    for event_id in result.event_ids:
        count = len(result.event_pairs(event_id))
        if count < min_pairs:
            result.unusable_events.add(event_id)
            issues.append(
                IngestIssue(
                    "insufficient_pairs",
                    "error",
                    f"only {count} usable P–S pair(s); a Wadati fit needs at least "
                    f"{min_pairs}. Add stations or lower min_pairs",
                    event_id,
                )
            )
    return result


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
_LONG_TIME = {"time", "pick_time", "arrival_time", "phase_time"}
_LONG_TRAVEL = {"travel_time", "tt", "traveltime"}
_PHASE = {"phase", "phase_hint", "phase_label"}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _first(row: dict[str, str], names: set[str] | list[str]) -> str:
    for name in names:
        if name in row and _clean(row[name]):
            return _clean(row[name])
    return ""


def load_csv_text(
    text: str,
    source: str = "csv:<text>",
    min_pairs: int = MIN_PAIRS_PER_EVENT,
) -> IngestResult:
    """Ingest a CSV phase-pick table given as text.

    Accepts the long layout (``phase`` + ``time``/``travel_time``) and the wide
    layout (``p_time``/``s_time`` or ``p_travel_time`` + ``s_minus_p``).
    """
    issues: list[IngestIssue] = []
    reader = csv.DictReader(io.StringIO(text.lstrip()))
    if not reader.fieldnames:
        issues.append(
            IngestIssue(
                "malformed_time", "error", "the CSV file has no header row"
            )
        )
        return IngestResult(source=source, issues=issues)

    header = {(name or "").strip().lower() for name in reader.fieldnames}
    for required in ("event_id", "station_id"):
        if required not in header:
            issues.append(
                IngestIssue(
                    "malformed_time",
                    "error",
                    f"required column {required!r} is missing; expected one of the "
                    "layouts documented in SCHEMA_DOC",
                )
            )
    if issues:
        return IngestResult(source=source, issues=issues)

    is_long = bool(header & _PHASE)
    records: list[_Record] = []

    for line_no, raw in enumerate(reader, start=2):
        row = {
            (k or "").strip().lower(): (v if v is not None else "")
            for k, v in raw.items()
        }
        event_id = _first(row, ["event_id"])
        station_id = _first(row, ["station_id", "station"])
        if not event_id or not station_id:
            issues.append(
                IngestIssue(
                    "malformed_time",
                    "error",
                    f"row {line_no}: event_id and station_id are both required",
                )
            )
            continue
        network = _first(row, ["network", "net"])
        channel = _first(row, ["channel", "cha"])

        origin_time: float | None = None
        origin_raw = _first(row, ["origin_time", "event_time"])
        if origin_raw:
            try:
                origin_time = parse_timestamp(origin_raw)
            except ValueError as exc:
                issues.append(
                    IngestIssue(
                        "malformed_time",
                        "error",
                        f"row {line_no}: origin_time — {exc}",
                        event_id,
                        station_id,
                    )
                )

        def add(phase: str, time_raw: str, travel_raw: str) -> None:
            time_value: float | None = None
            travel_value: float | None = None
            if time_raw:
                try:
                    time_value = parse_timestamp(time_raw)
                except ValueError as exc:
                    issues.append(
                        IngestIssue(
                            "malformed_time",
                            "error",
                            f"row {line_no}: {phase} time — {exc}",
                            event_id,
                            station_id,
                        )
                    )
                    return
            if travel_raw:
                try:
                    travel_value = parse_number(travel_raw)
                except ValueError as exc:
                    issues.append(
                        IngestIssue(
                            "malformed_time",
                            "error",
                            f"row {line_no}: {phase} travel time — {exc}",
                            event_id,
                            station_id,
                        )
                    )
                    return
            if time_value is None and travel_value is None:
                issues.append(
                    IngestIssue(
                        "malformed_time",
                        "error",
                        f"row {line_no}: {phase} pick has neither a timestamp nor a "
                        "travel time",
                        event_id,
                        station_id,
                    )
                )
                return
            records.append(
                _Record(
                    event_id=event_id,
                    station_id=station_id,
                    phase=phase,
                    time=time_value,
                    travel_time=travel_value,
                    origin_time=origin_time,
                    network=network,
                    channel=channel,
                    row=line_no,
                )
            )

        if is_long:
            phase = normalize_phase(_first(row, _PHASE))
            if phase is None:
                issues.append(
                    IngestIssue(
                        "unknown_phase",
                        "warning",
                        f"row {line_no}: phase {_first(row, _PHASE)!r} is neither a P "
                        "nor an S alias; the row is ignored",
                        event_id,
                        station_id,
                    )
                )
                continue
            add(phase, _first(row, _LONG_TIME), _first(row, _LONG_TRAVEL))
            continue

        # wide layout
        p_time_raw = _first(row, ["p_time", "tp", "p_arrival"])
        s_time_raw = _first(row, ["s_time", "ts", "s_arrival"])
        p_tt_raw = _first(row, ["p_travel_time", "ts_p", "tp_travel"])
        s_tt_raw = _first(row, ["s_travel_time", "ts_s"])
        s_minus_p_raw = _first(row, ["s_minus_p", "sp_interval", "ts_minus_tp"])

        if not s_tt_raw and p_tt_raw and s_minus_p_raw:
            try:
                s_tt_raw = str(
                    parse_number(p_tt_raw) + parse_number(s_minus_p_raw)
                )
            except ValueError as exc:
                issues.append(
                    IngestIssue(
                        "malformed_time",
                        "error",
                        f"row {line_no}: p_travel_time / s_minus_p — {exc}",
                        event_id,
                        station_id,
                    )
                )
                continue
        if not any([p_time_raw, s_time_raw, p_tt_raw, s_tt_raw]):
            issues.append(
                IngestIssue(
                    "malformed_time",
                    "error",
                    f"row {line_no}: no recognisable pick columns; expected p_time/"
                    "s_time or p_travel_time/s_minus_p",
                    event_id,
                    station_id,
                )
            )
            continue
        if p_time_raw or p_tt_raw:
            add("P", p_time_raw, p_tt_raw)
        else:
            issues.append(
                IngestIssue(
                    "missing_p",
                    "error",
                    f"row {line_no}: no P pick column value; a Wadati point needs both "
                    "phases",
                    event_id,
                    station_id,
                )
            )
        if s_time_raw or s_tt_raw:
            add("S", s_time_raw, s_tt_raw)
        else:
            issues.append(
                IngestIssue(
                    "missing_s",
                    "error",
                    f"row {line_no}: no S pick column value; a Wadati point needs both "
                    "phases",
                    event_id,
                    station_id,
                )
            )

    return _normalize(records, source, issues, min_pairs)


def load_csv_picks(
    path: str | Path,
    min_pairs: int = MIN_PAIRS_PER_EVENT,
) -> IngestResult:
    """Ingest a CSV phase-pick table from disk."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as e:
        logging.exception(f"Error: {e}")
        return IngestResult(
            source=f"csv:{target.name}",
            issues=[
                IngestIssue(
                    "malformed_time",
                    "error",
                    f"could not read {target}: {e}. Check the path and permissions",
                )
            ],
        )
    return load_csv_text(text, source=f"csv:{target.name}", min_pairs=min_pairs)


# ---------------------------------------------------------------------------
# QuakeML
# ---------------------------------------------------------------------------
def load_quakeml_picks(
    path: str | Path,
    min_pairs: int = MIN_PAIRS_PER_EVENT,
) -> IngestResult:
    """Ingest an ObsPy-readable QuakeML catalog.

    P travel times are measured against the preferred origin, falling back to
    the first available origin that carries a time.
    """
    target = Path(path)
    source = f"quakeml:{target.name}"
    issues: list[IngestIssue] = []
    try:
        from obspy import read_events
    except Exception as e:  # pragma: no cover - obspy is a hard dependency here
        logging.exception(f"Error: {e}")
        issues.append(
            IngestIssue(
                "malformed_time",
                "error",
                f"ObsPy is required to read QuakeML ({e}); install obspy>=1.5",
            )
        )
        return IngestResult(source=source, issues=issues)

    try:
        catalog = read_events(target, format="quakeml")
    except Exception as e:
        logging.exception(f"Error: {e}")
        issues.append(
            IngestIssue(
                "malformed_time",
                "error",
                f"ObsPy could not read {target} as QuakeML: {e}. Read it explicitly "
                'with read_events(path, format="quakeml") to see the full traceback',
            )
        )
        return IngestResult(source=source, issues=issues)

    records: list[_Record] = []
    for index, event in enumerate(catalog, start=1):
        event_id = str(event.resource_id).rsplit("/", 1)[-1] or f"event-{index}"

        origin = None
        try:
            origin = event.preferred_origin()
        except Exception:  # pragma: no cover - defensive
            logging.exception("Unexpected error")
            origin = None
        if origin is None or origin.time is None:
            origin = next(
                (o for o in event.origins if o.time is not None), origin
            )
        origin_time: float | None = None
        if origin is not None and origin.time is not None:
            origin_time = float(origin.time.timestamp)
        else:
            issues.append(
                IngestIssue(
                    "missing_origin_time",
                    "error",
                    "no origin carries a time, so origin-relative P travel times "
                    "cannot be derived; add an origin time or supply travel times "
                    "through the CSV path",
                    event_id,
                )
            )

        arrival_phase: dict[str, str] = {}
        if origin is not None:
            for arrival in origin.arrivals:
                if arrival.pick_id is not None and arrival.phase:
                    arrival_phase[str(arrival.pick_id)] = str(arrival.phase)

        if not event.picks:
            issues.append(
                IngestIssue(
                    "missing_p",
                    "error",
                    "event carries no picks at all; nothing to normalize",
                    event_id,
                )
            )
            continue

        for pick_no, pick in enumerate(event.picks, start=1):
            label = pick.phase_hint or arrival_phase.get(pick.resource_id, "")
            phase = normalize_phase(label)
            waveform = pick.waveform_id
            station_id = _clean(getattr(waveform, "station_code", "")) or "?"
            if station_id == "?":
                issues.append(
                    IngestIssue(
                        "unknown_phase",
                        "warning",
                        f"pick {pick_no} has no station code in its waveform_id and is "
                        "ignored",
                        event_id,
                    )
                )
                continue
            if phase is None:
                issues.append(
                    IngestIssue(
                        "unknown_phase",
                        "warning",
                        f"pick {pick_no} has phase label {label!r}, neither a P nor an "
                        "S alias; the pick is ignored",
                        event_id,
                        station_id,
                    )
                )
                continue
            if pick.time is None:
                issues.append(
                    IngestIssue(
                        "malformed_time",
                        "error",
                        f"{phase} pick {pick_no} has no time value",
                        event_id,
                        station_id,
                    )
                )
                continue
            records.append(
                _Record(
                    event_id=event_id,
                    station_id=station_id,
                    phase=phase,
                    time=float(pick.time.timestamp),
                    origin_time=origin_time,
                    network=_clean(getattr(waveform, "network_code", "")),
                    channel=_clean(getattr(waveform, "channel_code", "")),
                    row=pick_no,
                )
            )

    return _normalize(records, source, issues, min_pairs)


def load_picks(
    path: str | Path, min_pairs: int = MIN_PAIRS_PER_EVENT
) -> IngestResult:
    """Dispatch on file suffix: ``.csv`` -> CSV loader, ``.xml``/``.quakeml`` -> QuakeML."""
    target = Path(path)
    if target.suffix.lower() in {".xml", ".quakeml", ".qml"}:
        return load_quakeml_picks(target, min_pairs=min_pairs)
    return load_csv_picks(target, min_pairs=min_pairs)


def write_canonical_csv(result: IngestResult, path: str | Path) -> Path:
    """Write the normalized canonical table to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(result.as_rows())
    return target

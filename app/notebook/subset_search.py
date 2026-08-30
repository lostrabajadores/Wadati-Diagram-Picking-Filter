"""Exhaustive Wadati subset search — step 3 of the QC workflow.

Scientific statement of the algorithm
-------------------------------------
For one event, every station that carries a valid P and S pick contributes one
Wadati point ``(t_P, t_S - t_P)``.  In a homogeneous medium those points lie on
a straight line whose slope is ``Vp/Vs - 1``, so:

1. Fit ``s_minus_p`` against ``p_travel_time`` with
   :func:`scipy.stats.linregress` over **all** valid station pairs.
2. If the Pearson correlation reaches ``min_correlation`` (default **0.9**),
   the retained station count reaches ``min_stations`` and the implied
   ``Vp/Vs`` lies inside the optional ``[vp_vs_min, vp_vs_max]`` bounds, then
   **accept every pick** — no station is removed.
3. Otherwise enumerate subsets by removing **one** point, then **two**, then
   three, and so on — i.e. larger subsets are always tested before smaller
   ones.  Stop at the **first removal depth** that produces at least one
   qualifying candidate.
4. Among the candidates at that depth choose deterministically: highest
   Pearson ``|r|``, then lowest residual error (RMSE), then the stable
   station ordering of the removed stations.
5. Reject the event when no qualifying subset remains at or above
   ``min_stations`` (which is never allowed below
   :data:`ABSOLUTE_MIN_STATIONS` = 3, since a two-point line is trivially
   perfect and carries no information).

Rejection reasons are explicit and distinguish the two scientifically
different failures:

``rejected_correlation``
    no subset of sufficient size ever reached ``min_correlation`` — the picks
    themselves are inconsistent.
``rejected_vp_vs``
    a subset *did* satisfy the correlation and station-count requirements, but
    every such subset implies a ``Vp/Vs`` outside the configured bounds — the
    picks are internally consistent yet the velocity ratio is anomalous.
``rejected_insufficient_stations``
    the event never had ``min_stations`` valid pairs to begin with.
``rejected_search_truncated``
    the combination budget (``max_combinations``) was exhausted before a
    qualifying subset was found.

Everything here is plain Python plus NumPy/SciPy — importable, testable and
reusable outside the notebook.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import linregress

#: A Wadati line through fewer than three stations is meaningless, so the
#: configurable minimum station count is clamped up to this floor.
ABSOLUTE_MIN_STATIONS: int = 3

#: Default correlation threshold required to accept a subset.
DEFAULT_MIN_CORRELATION: float = 0.9

#: Default minimum number of retained stations.
DEFAULT_MIN_STATIONS: int = 4

#: Default (optional) Vp/Vs bounds — a wide but physically plausible crustal
#: window.  Set either bound to ``None`` to disable that side of the check.
DEFAULT_VP_VS_MIN: float = 1.50
DEFAULT_VP_VS_MAX: float = 2.10

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

REASON_ALL_PASS = "accepted_all_picks"
REASON_SUBSET = "accepted_after_removal"
REASON_CORRELATION = "rejected_correlation"
REASON_VP_VS = "rejected_vp_vs"
REASON_MIN_STATIONS = "rejected_insufficient_stations"
REASON_TRUNCATED = "rejected_search_truncated"

REASON_TEXT: dict[str, str] = {
    REASON_ALL_PASS: (
        "every valid station pair already satisfies the correlation, station "
        "count and Vp/Vs requirements; nothing was removed"
    ),
    REASON_SUBSET: (
        "the full set failed, but a qualifying subset was found at the first "
        "removal depth that produced candidates"
    ),
    REASON_CORRELATION: (
        "no subset at or above the minimum station count reached the required "
        "correlation — the P/S picks are mutually inconsistent"
    ),
    REASON_VP_VS: (
        "a subset satisfied the correlation and station-count requirements, "
        "but its implied Vp/Vs is outside the configured bounds — an "
        "anomalous velocity ratio, not a correlation failure"
    ),
    REASON_MIN_STATIONS: (
        "the event carries fewer valid P–S station pairs than the configured "
        "minimum station count"
    ),
    REASON_TRUNCATED: (
        "the combination budget was exhausted before a qualifying subset was "
        "found; raise max_combinations or max_removals to search further"
    ),
}

EVENT_QC_COLUMNS: list[str] = [
    "event_id",
    "source",
    "status",
    "reason",
    "reason_detail",
    "original_stations",
    "retained_stations",
    "removed_count",
    "removed_stations",
    "search_depth",
    "max_depth_searched",
    "combinations_evaluated",
    "search_truncated",
    "slope",
    "intercept",
    "pearson_r",
    "r_squared",
    "p_value",
    "slope_stderr",
    "intercept_stderr",
    "rmse",
    "mae",
    "max_abs_residual",
    "residual_std",
    "vp_vs",
    "vp_vs_stderr",
    "initial_pearson_r",
    "initial_vp_vs",
    "min_correlation",
    "min_stations",
    "vp_vs_min",
    "vp_vs_max",
]

PICK_QC_COLUMNS: list[str] = [
    "event_id",
    "station_id",
    "network",
    "channel",
    "p_travel_time",
    "s_minus_p",
    "state",
    "residual",
    "abs_residual",
    "apparent_vp_vs",
    "event_status",
    "event_reason",
]

CONFIG_DOC = """\
SUBSET SEARCH CONFIGURATION
---------------------------
  min_correlation  float   default 0.90   minimum Pearson |r| a subset must reach
  min_stations     int     default 4      minimum retained stations (clamped up to 3)
  vp_vs_min        float?  default 1.50   lower Vp/Vs bound (None disables)
  vp_vs_max        float?  default 2.10   upper Vp/Vs bound (None disables)
  max_removals     int?    default None   deepest removal depth (None = down to min_stations)
  max_combinations int     default 200000 evaluation budget per event

EXAMPLES
--------
  SearchConfig()                                    # published defaults
  SearchConfig(min_correlation=0.95)                # stricter correlation
  SearchConfig(min_stations=6)                      # demand six stations
  SearchConfig(vp_vs_min=None, vp_vs_max=None)      # correlation only
  SearchConfig(vp_vs_min=1.60, vp_vs_max=1.85)      # narrow crustal window
  SearchConfig(max_removals=2)                      # remove at most two picks
  SearchConfig(min_stations=2)                      # clamped to 3 with a warning

STATUS / REASON CODES
---------------------
  accepted   accepted_all_picks            nothing removed
  accepted   accepted_after_removal        qualifying subset found
  rejected   rejected_correlation          correlation never reached
  rejected   rejected_vp_vs                anomalous Vp/Vs (correlation was fine)
  rejected   rejected_insufficient_stations too few valid pairs to begin with
  rejected   rejected_search_truncated     combination budget exhausted

RESULT API
----------
  EventQC.status / reason / reason_detail
  EventQC.fit                  FitStats of the retained subset (or the full set)
  EventQC.initial_fit          FitStats of every valid pair
  EventQC.retained_stations / removed_stations / outlier_stations
  EventQC.search_depth / max_depth_searched / combinations_evaluated
  EventQC.as_row()             one flat QC row  (EVENT_QC_COLUMNS)
  EventQC.pick_rows()          one row per pick (PICK_QC_COLUMNS)
  EventQC.report()             printable per-event report

  DatasetQC.results / accepted / rejected / summary() / report()
  DatasetQC.event_rows() / pick_rows() / event_dataframe() / pick_dataframe()
  write_event_qc_csv(dataset, path) / write_pick_qc_csv(dataset, path)
  event_qc_csv_text(dataset) / pick_qc_csv_text(dataset)   (download widgets)
"""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchConfig:
    """Configurable acceptance criteria for the exhaustive subset search."""

    min_correlation: float = DEFAULT_MIN_CORRELATION
    min_stations: int = DEFAULT_MIN_STATIONS
    vp_vs_min: float | None = DEFAULT_VP_VS_MIN
    vp_vs_max: float | None = DEFAULT_VP_VS_MAX
    max_removals: int | None = None
    max_combinations: int = 200_000

    def normalized(self) -> tuple["SearchConfig", list[str]]:
        """Return a validated copy plus any adjustment notes."""
        notes: list[str] = []
        min_stations = int(self.min_stations)
        if min_stations < ABSOLUTE_MIN_STATIONS:
            notes.append(
                f"min_stations={min_stations} raised to the absolute floor "
                f"{ABSOLUTE_MIN_STATIONS}: a Wadati line through fewer than "
                "three stations carries no information"
            )
            min_stations = ABSOLUTE_MIN_STATIONS
        correlation = float(self.min_correlation)
        if not 0.0 <= correlation <= 1.0:
            notes.append(f"min_correlation={correlation} clamped into [0, 1]")
            correlation = min(max(correlation, 0.0), 1.0)
        lo, hi = self.vp_vs_min, self.vp_vs_max
        if lo is not None and hi is not None and float(lo) > float(hi):
            notes.append(
                f"vp_vs bounds were reversed ({lo} > {hi}) and have been swapped"
            )
            lo, hi = hi, lo
        return (
            replace(
                self,
                min_correlation=correlation,
                min_stations=min_stations,
                vp_vs_min=None if lo is None else float(lo),
                vp_vs_max=None if hi is None else float(hi),
                max_combinations=max(1, int(self.max_combinations)),
            ),
            notes,
        )

    def describe(self) -> str:
        bounds = (
            "disabled"
            if self.vp_vs_min is None and self.vp_vs_max is None
            else f"{self.vp_vs_min if self.vp_vs_min is not None else '-inf'}"
            f" .. {self.vp_vs_max if self.vp_vs_max is not None else '+inf'}"
        )
        depth = (
            "down to min_stations"
            if self.max_removals is None
            else f"at most {self.max_removals} removal(s)"
        )
        return (
            f"min |r| >= {self.min_correlation:.2f} · "
            f"min stations {self.min_stations} · Vp/Vs {bounds} · {depth} · "
            f"budget {self.max_combinations:,} combinations"
        )


# ---------------------------------------------------------------------------
# points and fits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WadatiPoint:
    """One station's Wadati point for a single event."""

    station_id: str
    p_travel_time: float
    s_minus_p: float
    network: str = ""
    channel: str = ""

    @property
    def apparent_vp_vs(self) -> float:
        if self.p_travel_time <= 0.0:
            return float("nan")
        return 1.0 + self.s_minus_p / self.p_travel_time


def points_from_pairs(pairs) -> list[WadatiPoint]:
    """Convert canonical :class:`ingest.PickPair` objects to Wadati points.

    The order is stable and scientifically meaningful: increasing P travel
    time, then station code — so every deterministic tie-break downstream is
    reproducible.
    """
    points = [
        WadatiPoint(
            station_id=str(pair.station_id),
            p_travel_time=float(pair.p_travel_time),
            s_minus_p=float(pair.s_minus_p),
            network=str(getattr(pair, "network", "") or ""),
            channel=str(getattr(pair, "channel", "") or ""),
        )
        for pair in pairs
    ]
    points.sort(key=lambda p: (p.p_travel_time, p.station_id))
    return points


@dataclass(frozen=True)
class FitStats:
    """Least-squares Wadati regression statistics for one subset."""

    n: int
    slope: float
    intercept: float
    pearson_r: float
    r_squared: float
    p_value: float
    slope_stderr: float
    intercept_stderr: float
    rmse: float
    mae: float
    max_abs_residual: float
    residual_std: float
    vp_vs: float
    vp_vs_stderr: float
    stations: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.n >= ABSOLUTE_MIN_STATIONS and math.isfinite(self.pearson_r)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 6),
            "pearson_r": round(self.pearson_r, 6),
            "r_squared": round(self.r_squared, 6),
            "p_value": self.p_value,
            "slope_stderr": round(self.slope_stderr, 6),
            "intercept_stderr": round(self.intercept_stderr, 6),
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "max_abs_residual": round(self.max_abs_residual, 6),
            "residual_std": round(self.residual_std, 6),
            "vp_vs": round(self.vp_vs, 6),
            "vp_vs_stderr": round(self.vp_vs_stderr, 6),
        }


NULL_FIT = FitStats(
    n=0,
    slope=float("nan"),
    intercept=float("nan"),
    pearson_r=float("nan"),
    r_squared=float("nan"),
    p_value=float("nan"),
    slope_stderr=float("nan"),
    intercept_stderr=float("nan"),
    rmse=float("nan"),
    mae=float("nan"),
    max_abs_residual=float("nan"),
    residual_std=float("nan"),
    vp_vs=float("nan"),
    vp_vs_stderr=float("nan"),
)


def fit_subset(points: list[WadatiPoint]) -> FitStats:
    """Fit S−P against P travel time with :func:`scipy.stats.linregress`."""
    if len(points) < ABSOLUTE_MIN_STATIONS:
        return replace(
            NULL_FIT,
            n=len(points),
            stations=tuple(p.station_id for p in points),
        )
    x = np.array([p.p_travel_time for p in points], dtype=float)
    y = np.array([p.s_minus_p for p in points], dtype=float)
    if float(np.ptp(x)) == 0.0:
        return replace(
            NULL_FIT,
            n=len(points),
            stations=tuple(p.station_id for p in points),
        )
    fit = linregress(x, y)
    predicted = fit.intercept + fit.slope * x
    residual = y - predicted
    dof = max(len(points) - 2, 1)
    return FitStats(
        n=len(points),
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        pearson_r=float(fit.rvalue),
        r_squared=float(fit.rvalue) ** 2,
        p_value=float(fit.pvalue),
        slope_stderr=float(fit.stderr),
        intercept_stderr=float(fit.intercept_stderr),
        rmse=float(np.sqrt(float(np.mean(residual**2)))),
        mae=float(np.mean(np.abs(residual))),
        max_abs_residual=float(np.max(np.abs(residual))),
        residual_std=float(np.sqrt(float(np.sum(residual**2)) / dof)),
        vp_vs=float(fit.slope) + 1.0,
        vp_vs_stderr=float(fit.stderr),
        stations=tuple(p.station_id for p in points),
    )


def residuals(points: list[WadatiPoint], fit: FitStats) -> list[float]:
    """Signed residuals of ``points`` against ``fit`` (empty when unfittable)."""
    if not fit.valid:
        return [float("nan")] * len(points)
    return [
        p.s_minus_p - (fit.intercept + fit.slope * p.p_travel_time)
        for p in points
    ]


def correlation_ok(fit: FitStats, config: SearchConfig) -> bool:
    return fit.valid and abs(fit.pearson_r) >= config.min_correlation


def vp_vs_ok(fit: FitStats, config: SearchConfig) -> bool:
    if not fit.valid or not math.isfinite(fit.vp_vs):
        return False
    if config.vp_vs_min is not None and fit.vp_vs < config.vp_vs_min:
        return False
    if config.vp_vs_max is not None and fit.vp_vs > config.vp_vs_max:
        return False
    return True


# ---------------------------------------------------------------------------
# per-event result
# ---------------------------------------------------------------------------
@dataclass
class EventQC:
    """Everything the subset search determined about one event."""

    event_id: str
    source: str
    config: SearchConfig
    points: list[WadatiPoint]
    status: str
    reason: str
    retained: list[WadatiPoint] = field(default_factory=list)
    removed: list[WadatiPoint] = field(default_factory=list)
    fit: FitStats = NULL_FIT
    initial_fit: FitStats = NULL_FIT
    search_depth: int = 0
    max_depth_searched: int = 0
    combinations_evaluated: int = 0
    search_truncated: bool = False
    notes: list[str] = field(default_factory=list)

    # --- views -----------------------------------------------------------
    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @property
    def reason_detail(self) -> str:
        return REASON_TEXT.get(self.reason, self.reason)

    @property
    def original_stations(self) -> int:
        return len(self.points)

    @property
    def retained_stations(self) -> int:
        return len(self.retained)

    @property
    def retained_station_ids(self) -> list[str]:
        return [p.station_id for p in self.retained]

    @property
    def removed_stations(self) -> list[str]:
        return [p.station_id for p in self.removed]

    #: The removed stations are exactly the picks flagged as outliers.
    @property
    def outlier_stations(self) -> list[str]:
        return self.removed_stations

    @property
    def vp_vs(self) -> float:
        return self.fit.vp_vs

    def as_row(self) -> dict[str, str | float | int | bool]:
        fit = self.fit if self.fit.valid else self.initial_fit
        return {
            "event_id": self.event_id,
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "original_stations": self.original_stations,
            "retained_stations": self.retained_stations,
            "removed_count": len(self.removed),
            "removed_stations": ";".join(self.removed_stations),
            "search_depth": self.search_depth,
            "max_depth_searched": self.max_depth_searched,
            "combinations_evaluated": self.combinations_evaluated,
            "search_truncated": self.search_truncated,
            "slope": _r(fit.slope),
            "intercept": _r(fit.intercept),
            "pearson_r": _r(fit.pearson_r, 6),
            "r_squared": _r(fit.r_squared, 6),
            "p_value": _r(fit.p_value, 8),
            "slope_stderr": _r(fit.slope_stderr, 6),
            "intercept_stderr": _r(fit.intercept_stderr, 6),
            "rmse": _r(fit.rmse),
            "mae": _r(fit.mae),
            "max_abs_residual": _r(fit.max_abs_residual),
            "residual_std": _r(fit.residual_std),
            "vp_vs": _r(fit.vp_vs),
            "vp_vs_stderr": _r(fit.vp_vs_stderr),
            "initial_pearson_r": _r(self.initial_fit.pearson_r, 6),
            "initial_vp_vs": _r(self.initial_fit.vp_vs),
            "min_correlation": self.config.min_correlation,
            "min_stations": self.config.min_stations,
            "vp_vs_min": ""
            if self.config.vp_vs_min is None
            else self.config.vp_vs_min,
            "vp_vs_max": ""
            if self.config.vp_vs_max is None
            else self.config.vp_vs_max,
        }

    def pick_rows(self) -> list[dict[str, str | float]]:
        """One row per pick, marked ``retained`` or ``outlier``."""
        reference = self.fit if self.fit.valid else self.initial_fit
        removed = set(self.removed_stations)
        rows: list[dict[str, str | float]] = []
        for point in self.points:
            if reference.valid:
                residual = point.s_minus_p - (
                    reference.intercept + reference.slope * point.p_travel_time
                )
            else:
                residual = float("nan")
            rows.append(
                {
                    "event_id": self.event_id,
                    "station_id": point.station_id,
                    "network": point.network,
                    "channel": point.channel,
                    "p_travel_time": _r(point.p_travel_time),
                    "s_minus_p": _r(point.s_minus_p),
                    "state": "outlier"
                    if point.station_id in removed
                    else "retained",
                    "residual": _r(residual),
                    "abs_residual": _r(abs(residual)),
                    "apparent_vp_vs": _r(point.apparent_vp_vs),
                    "event_status": self.status,
                    "event_reason": self.reason,
                }
            )
        return rows

    def plot_frame(self):  # pragma: no cover - notebook convenience
        """``station / ts_p / s_minus_p / rejected`` frame for the Wadati plot."""
        import pandas as pd

        removed = set(self.removed_stations)
        return pd.DataFrame(
            [
                {
                    "station": p.station_id,
                    "ts_p": p.p_travel_time,
                    "s_minus_p": p.s_minus_p,
                    "rejected": p.station_id in removed,
                }
                for p in self.points
            ]
        )

    def plot_fit(self) -> dict[str, float]:  # pragma: no cover - convenience
        """``slope / intercept / r / vp_vs / stderr`` dict for the plot helper."""
        fit = self.fit if self.fit.valid else self.initial_fit
        return {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r": fit.pearson_r,
            "vp_vs": fit.vp_vs,
            "stderr": fit.slope_stderr,
        }

    def report(self) -> str:
        fit = self.fit if self.fit.valid else self.initial_fit
        lines = [
            f"event            {self.event_id}   [{self.status.upper()}]",
            f"reason           {self.reason} — {self.reason_detail}",
            f"criteria         {self.config.describe()}",
            f"stations         {self.retained_stations} retained of "
            f"{self.original_stations}"
            + (
                f"   removed: {', '.join(self.removed_stations)}"
                if self.removed
                else ""
            ),
            f"search           depth {self.search_depth} of "
            f"{self.max_depth_searched} · "
            f"{self.combinations_evaluated:,} combination(s) evaluated"
            + ("  [TRUNCATED]" if self.search_truncated else ""),
        ]
        if fit.valid:
            lines += [
                f"regression       S−P = {fit.slope:.4f}·t_P + {fit.intercept:.4f}",
                f"correlation      r = {fit.pearson_r:.4f}   r² = {fit.r_squared:.4f}"
                f"   p = {fit.p_value:.3e}",
                f"standard errors  slope ± {fit.slope_stderr:.4f}   "
                f"intercept ± {fit.intercept_stderr:.4f}",
                f"residuals        rmse {fit.rmse:.4f}   mae {fit.mae:.4f}   "
                f"max |r| {fit.max_abs_residual:.4f}   s {fit.residual_std:.4f}",
                f"velocity ratio   Vp/Vs = {fit.vp_vs:.4f} ± {fit.vp_vs_stderr:.4f}",
                f"initial fit      r = {self.initial_fit.pearson_r:.4f}   "
                f"Vp/Vs = {self.initial_fit.vp_vs:.4f}  (all "
                f"{self.initial_fit.n} pairs)",
            ]
        else:
            lines.append(
                "regression       not computable — fewer than "
                f"{ABSOLUTE_MIN_STATIONS} usable pairs"
            )
        lines.extend(f"note             {note}" for note in self.notes)
        return "\n".join(lines)


def _r(value: float, digits: int = 4) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(number):
        return float("nan")
    return round(number, digits)


# ---------------------------------------------------------------------------
# the search itself
# ---------------------------------------------------------------------------
def search_event(
    event_id: str,
    points: list[WadatiPoint],
    config: SearchConfig | None = None,
    source: str = "",
) -> EventQC:
    """Run the exhaustive Wadati subset search for a single event.

    Larger subsets are always evaluated before smaller ones, the search stops
    at the first removal depth that yields qualifying candidates, and the
    winner is chosen by highest ``|r|``, then lowest RMSE, then the stable
    station ordering of the removed stations.
    """
    config, notes = (config or SearchConfig()).normalized()
    ordered = sorted(points, key=lambda p: (p.p_travel_time, p.station_id))
    initial = fit_subset(ordered)

    base = EventQC(
        event_id=event_id,
        source=source,
        config=config,
        points=ordered,
        status=STATUS_REJECTED,
        reason=REASON_MIN_STATIONS,
        initial_fit=initial,
        notes=notes,
    )

    if len(ordered) < config.min_stations:
        base.fit = initial
        return base

    # --- depth 0: accept every pick when it already qualifies -------------
    evaluated = 1
    if correlation_ok(initial, config) and vp_vs_ok(initial, config):
        base.status = STATUS_ACCEPTED
        base.reason = REASON_ALL_PASS
        base.retained = list(ordered)
        base.fit = initial
        base.combinations_evaluated = evaluated
        return base

    correlation_ever_ok = correlation_ok(initial, config)
    max_removals = len(ordered) - config.min_stations
    if config.max_removals is not None:
        max_removals = min(max_removals, max(0, int(config.max_removals)))

    truncated = False
    depth_reached = 0
    for depth in range(1, max_removals + 1):
        depth_reached = depth
        candidates: list[tuple[float, float, tuple[str, ...], list[int]]] = []
        for removal in combinations(range(len(ordered)), depth):
            if evaluated >= config.max_combinations:
                truncated = True
                break
            evaluated += 1
            keep_index = [i for i in range(len(ordered)) if i not in removal]
            subset = [ordered[i] for i in keep_index]
            fit = fit_subset(subset)
            if not correlation_ok(fit, config):
                continue
            correlation_ever_ok = True
            if not vp_vs_ok(fit, config):
                continue
            removed_ids = tuple(ordered[i].station_id for i in removal)
            candidates.append(
                (-abs(fit.pearson_r), fit.rmse, removed_ids, keep_index)
            )
        if candidates:
            candidates.sort()
            _, _, removed_ids, keep_index = candidates[0]
            keep = set(keep_index)
            base.status = STATUS_ACCEPTED
            base.reason = REASON_SUBSET
            base.retained = [ordered[i] for i in keep_index]
            base.removed = [
                ordered[i] for i in range(len(ordered)) if i not in keep
            ]
            base.fit = fit_subset(base.retained)
            base.search_depth = depth
            base.max_depth_searched = depth
            base.combinations_evaluated = evaluated
            return base
        if truncated:
            break

    base.max_depth_searched = depth_reached
    base.combinations_evaluated = evaluated
    base.search_truncated = truncated
    base.retained = []
    base.fit = initial
    if truncated:
        base.reason = REASON_TRUNCATED
    elif correlation_ever_ok:
        base.reason = REASON_VP_VS
    else:
        base.reason = REASON_CORRELATION
    return base


# ---------------------------------------------------------------------------
# dataset-wide run
# ---------------------------------------------------------------------------
@dataclass
class DatasetQC:
    """Subset-search results for a whole dataset."""

    source: str
    config: SearchConfig
    results: list[EventQC] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def accepted(self) -> list[EventQC]:
        return [r for r in self.results if r.accepted]

    @property
    def rejected(self) -> list[EventQC]:
        return [r for r in self.results if not r.accepted]

    def by_id(self, event_id: str) -> EventQC | None:
        return next((r for r in self.results if r.event_id == event_id), None)

    @property
    def event_ids(self) -> list[str]:
        return [r.event_id for r in self.results]

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.reason] = counts.get(result.reason, 0) + 1
        return dict(sorted(counts.items()))

    def event_rows(self) -> list[dict[str, str | float | int | bool]]:
        return [r.as_row() for r in self.results]

    def pick_rows(self) -> list[dict[str, str | float]]:
        rows: list[dict[str, str | float]] = []
        for result in self.results:
            rows.extend(result.pick_rows())
        return rows

    def summary(self) -> dict[str, str | float | int]:
        accepted = self.accepted
        retained = sum(r.retained_stations for r in accepted)
        original = sum(r.original_stations for r in self.results)
        removed = sum(len(r.removed) for r in self.results)
        ratios = [
            r.fit.vp_vs
            for r in accepted
            if r.fit.valid and math.isfinite(r.fit.vp_vs)
        ]
        return {
            "source": self.source,
            "events": len(self.results),
            "accepted": len(accepted),
            "rejected": len(self.rejected),
            "picks_total": original,
            "picks_retained": retained,
            "picks_removed": removed,
            "vp_vs_mean": _r(sum(ratios) / len(ratios))
            if ratios
            else float("nan"),
            "vp_vs_min": _r(min(ratios)) if ratios else float("nan"),
            "vp_vs_max": _r(max(ratios)) if ratios else float("nan"),
            "combinations_evaluated": sum(
                r.combinations_evaluated for r in self.results
            ),
        }

    def report(self) -> str:
        summary = self.summary()
        lines = [
            f"source            {self.source}",
            f"criteria          {self.config.describe()}",
            f"events            {summary['events']}  "
            f"(accepted {summary['accepted']}, rejected {summary['rejected']})",
            f"picks             {summary['picks_retained']} retained / "
            f"{summary['picks_removed']} removed of {summary['picks_total']}",
            f"Vp/Vs (accepted)  mean {summary['vp_vs_mean']}  "
            f"range {summary['vp_vs_min']} .. {summary['vp_vs_max']}",
            f"combinations      {summary['combinations_evaluated']:,} evaluated",
            "reasons           "
            + ", ".join(f"{k}={v}" for k, v in self.reason_counts().items()),
        ]
        if self.skipped:
            lines.append("")
            lines.extend(
                f"skipped {event_id}: {why}" for event_id, why in self.skipped
            )
        lines.append("")
        for result in self.results:
            mark = "OK " if result.accepted else "REJ"
            fit = result.fit if result.fit.valid else result.initial_fit
            lines.append(
                f"[{mark}] {result.event_id:<16} "
                f"n={result.retained_stations}/{result.original_stations} "
                f"r={fit.pearson_r:+.4f} Vp/Vs={fit.vp_vs:.3f} "
                f"depth={result.search_depth} {result.reason}"
            )
        return "\n".join(lines)

    def event_dataframe(self):  # pragma: no cover - notebook convenience
        import pandas as pd

        return pd.DataFrame(self.event_rows(), columns=EVENT_QC_COLUMNS)

    def pick_dataframe(self):  # pragma: no cover - notebook convenience
        import pandas as pd

        return pd.DataFrame(self.pick_rows(), columns=PICK_QC_COLUMNS)


def run_subset_search(
    ingest_result,
    config: SearchConfig | None = None,
    usable_only: bool = True,
) -> DatasetQC:
    """Run the search over every event of a canonical :class:`IngestResult`."""
    config, _ = (config or SearchConfig()).normalized()
    dataset = DatasetQC(
        source=getattr(ingest_result, "source", "<ingest>"), config=config
    )
    pairs = ingest_result.usable_pairs if usable_only else ingest_result.pairs
    grouped: dict[str, list] = {}
    order: list[str] = []
    for pair in pairs:
        if pair.event_id not in grouped:
            grouped[pair.event_id] = []
            order.append(pair.event_id)
        grouped[pair.event_id].append(pair)

    for event_id in order:
        points = points_from_pairs(grouped[event_id])
        if len(points) < ABSOLUTE_MIN_STATIONS:
            dataset.skipped.append(
                (
                    event_id,
                    f"only {len(points)} valid P–S pair(s); a Wadati fit needs at "
                    f"least {ABSOLUTE_MIN_STATIONS}",
                )
            )
            continue
        dataset.results.append(
            search_event(
                event_id,
                points,
                config,
                source=getattr(ingest_result, "source", ""),
            )
        )
    return dataset


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def _csv_text(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return buffer.getvalue()


def event_qc_csv_text(dataset: DatasetQC) -> str:
    """Detailed per-event QC table as CSV text (for download widgets)."""
    return _csv_text(dataset.event_rows(), EVENT_QC_COLUMNS)


def pick_qc_csv_text(dataset: DatasetQC) -> str:
    """Per-pick retained/outlier table as CSV text (for download widgets)."""
    return _csv_text(dataset.pick_rows(), PICK_QC_COLUMNS)


def write_event_qc_csv(dataset: DatasetQC, path: str | Path) -> Path:
    """Write the per-event QC table to ``path`` and return it."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(event_qc_csv_text(dataset), encoding="utf-8")
    except OSError as e:
        logging.exception(f"Error: {e}")
        raise
    return target


def write_pick_qc_csv(dataset: DatasetQC, path: str | Path) -> Path:
    """Write the per-pick retained/outlier table to ``path`` and return it."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pick_qc_csv_text(dataset), encoding="utf-8")
    except OSError as e:
        logging.exception(f"Error: {e}")
        raise
    return target

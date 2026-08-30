"""Worked examples / tests for the exhaustive Wadati subset search.

Four scientifically distinct situations are covered, each built
deterministically so the expectations are exact:

1. ``all_pass``          — every station already qualifies, nothing removed.
2. ``one_outlier``       — a single mis-picked S fixes the event at depth 1.
3. ``two_outliers``      — two mis-picks; depth 1 is not enough, depth 2 is.
4. ``anomalous_vp_vs``   — an excellent correlation with a Vp/Vs outside the
   configured bounds: rejected as ``rejected_vp_vs``, *not* as a correlation
   failure.
5. ``irrecoverable``     — sawtooth picks: no subset at or above the minimum
   station count ever reaches the correlation threshold.

Run as a script::

    python -m app.notebook.subset_examples
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # in-app import
    from app.notebook.subset_search import (
        REASON_ALL_PASS,
        REASON_CORRELATION,
        REASON_SUBSET,
        REASON_VP_VS,
        STATUS_ACCEPTED,
        STATUS_REJECTED,
        EventQC,
        SearchConfig,
        WadatiPoint,
        search_event,
    )
except ImportError:  # notebook-local copy
    from wadati_subset import (  # type: ignore[no-redef]
        REASON_ALL_PASS,
        REASON_CORRELATION,
        REASON_SUBSET,
        REASON_VP_VS,
        STATUS_ACCEPTED,
        STATUS_REJECTED,
        EventQC,
        SearchConfig,
        WadatiPoint,
        search_event,
    )

TRUE_VP_VS = 1.732
STATIONS = [
    "LOBH",
    "LOCA",
    "LODA2",
    "LODU1",
    "LOLU",
    "LOMC",
    "LONA2",
    "LONE3",
    "LOPU",
    "LORA",
    "LOSA",
    "LOTA",
]
#: Small deterministic pick jitter (s), cycled over the stations.
JITTER = [
    0.04,
    -0.06,
    0.02,
    0.07,
    -0.03,
    -0.05,
    0.06,
    0.01,
    -0.07,
    0.05,
    -0.02,
    0.03,
]


def _line_points(
    count: int,
    vp_vs: float = TRUE_VP_VS,
    outliers: dict[int, float] | None = None,
) -> list[WadatiPoint]:
    """``count`` collinear Wadati points, with optional S−P perturbations."""
    outliers = outliers or {}
    points: list[WadatiPoint] = []
    for index in range(count):
        p_travel = round(2.6 + 1.48 * index, 3)
        s_minus_p = (vp_vs - 1.0) * p_travel + JITTER[index % len(JITTER)]
        s_minus_p += outliers.get(index, 0.0)
        points.append(
            WadatiPoint(
                station_id=STATIONS[index % len(STATIONS)],
                p_travel_time=p_travel,
                s_minus_p=round(s_minus_p, 3),
                network="LO",
                channel="EHZ",
            )
        )
    return points


def _sawtooth_points(
    count: int = 8, amplitude: float = 9.0
) -> list[WadatiPoint]:
    """Alternating high/low S−P picks: no subset can be made collinear."""
    points: list[WadatiPoint] = []
    for index in range(count):
        p_travel = round(3.0 + 2.0 * index, 3)
        s_minus_p = (TRUE_VP_VS - 1.0) * p_travel + (
            amplitude if index % 2 == 0 else -amplitude
        )
        points.append(
            WadatiPoint(
                station_id=STATIONS[index % len(STATIONS)],
                p_travel_time=p_travel,
                s_minus_p=round(s_minus_p + 12.0, 3),
                network="LO",
                channel="EHZ",
            )
        )
    return points


@dataclass
class Example:
    """One named example with its configuration and expectations."""

    key: str
    title: str
    description: str
    points: list[WadatiPoint]
    config: SearchConfig
    expected_status: str
    expected_reason: str
    expected_removed: list[str] = field(default_factory=list)
    expected_depth: int = 0

    def run(self) -> EventQC:
        return search_event(
            self.key, self.points, self.config, source="example:subset_search"
        )


def examples() -> list[Example]:
    """The five worked examples, in narrative order."""
    return [
        Example(
            key="all_pass",
            title="Every station already qualifies",
            description=(
                "Eight collinear stations with sub-0.1 s pick jitter: the very "
                "first fit clears r ≥ 0.9, the station count and the Vp/Vs "
                "bounds, so no station is removed and the search never "
                "enumerates a subset."
            ),
            points=_line_points(8),
            config=SearchConfig(),
            expected_status=STATUS_ACCEPTED,
            expected_reason=REASON_ALL_PASS,
        ),
        Example(
            key="one_outlier",
            title="Fixed by removing one outlier",
            description=(
                "Twelve stations, one grossly late S pick (+9 s). The full set "
                "fails the correlation threshold; removing that single station "
                "at depth 1 restores the line."
            ),
            points=_line_points(12, outliers={4: 9.0}),
            config=SearchConfig(),
            expected_status=STATUS_ACCEPTED,
            expected_reason=REASON_SUBSET,
            expected_removed=[STATIONS[4]],
            expected_depth=1,
        ),
        Example(
            key="two_outliers",
            title="Fixed by removing two outliers",
            description=(
                "Twelve stations, one late (+9 s) and one early (−9 s) S pick. "
                "No single removal reaches r ≥ 0.9, so the search descends to "
                "depth 2 and stops there — larger subsets are always tested "
                "before smaller ones."
            ),
            points=_line_points(12, outliers={4: 9.0, 9: -9.0}),
            config=SearchConfig(),
            expected_status=STATUS_ACCEPTED,
            expected_reason=REASON_SUBSET,
            expected_removed=sorted([STATIONS[4], STATIONS[9]]),
            expected_depth=2,
        ),
        Example(
            key="anomalous_vp_vs",
            title="Anomalous Vp/Vs, not a correlation failure",
            description=(
                "Seven perfectly collinear stations implying Vp/Vs ≈ 2.62 — "
                "far outside the 1.50–2.10 window. The correlation criterion "
                "is met at every depth, so the event is rejected explicitly as "
                "rejected_vp_vs rather than rejected_correlation."
            ),
            points=_line_points(7, vp_vs=2.62),
            config=SearchConfig(),
            expected_status=STATUS_REJECTED,
            expected_reason=REASON_VP_VS,
        ),
        Example(
            key="irrecoverable",
            title="Irrecoverable — no qualifying subset",
            description=(
                "Eight stations whose S−P intervals alternate ±9 s around the "
                "trend. With min_stations raised to 6 the search may remove at "
                "most two picks, and no such subset reaches r ≥ 0.9, so the "
                "event is rejected for correlation."
            ),
            points=_sawtooth_points(8, amplitude=9.0),
            config=SearchConfig(min_stations=6),
            expected_status=STATUS_REJECTED,
            expected_reason=REASON_CORRELATION,
        ),
    ]


def run_examples() -> list[tuple[Example, EventQC]]:
    """Run every example and return ``(example, result)`` pairs."""
    return [(example, example.run()) for example in examples()]


def check_examples() -> list[dict[str, str | int | float | bool]]:
    """Run and assert every example; return a tidy outcome table."""
    rows: list[dict[str, str | int | float | bool]] = []
    for example, result in run_examples():
        assert result.status == example.expected_status, (
            f"{example.key}: expected status {example.expected_status}, "
            f"got {result.status} ({result.reason})"
        )
        assert result.reason == example.expected_reason, (
            f"{example.key}: expected reason {example.expected_reason}, "
            f"got {result.reason}"
        )
        assert result.search_depth == example.expected_depth, (
            f"{example.key}: expected search depth {example.expected_depth}, "
            f"got {result.search_depth}"
        )
        assert sorted(result.removed_stations) == sorted(
            example.expected_removed
        ), (
            f"{example.key}: expected removed {example.expected_removed}, "
            f"got {result.removed_stations}"
        )
        if result.accepted:
            assert abs(result.fit.pearson_r) >= example.config.min_correlation
            assert result.retained_stations >= example.config.min_stations
        rows.append(
            {
                "example": example.key,
                "title": example.title,
                "status": result.status,
                "reason": result.reason,
                "stations": f"{result.retained_stations}/{result.original_stations}",
                "removed": ", ".join(result.removed_stations) or "—",
                "depth": result.search_depth,
                "combinations": result.combinations_evaluated,
                "pearson_r": round(float(result.fit.pearson_r), 4),
                "vp_vs": round(float(result.fit.vp_vs), 3),
            }
        )
    return rows


def main() -> None:
    for example, result in run_examples():
        print("=" * 72)
        print(f"{example.key} — {example.title}")
        print(example.description)
        print("-" * 72)
        print(result.report())
        print()
    check_examples()
    print("All subset-search examples behaved exactly as documented.")


if __name__ == "__main__":
    main()

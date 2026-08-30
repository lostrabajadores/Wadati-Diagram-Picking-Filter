import reflex as rx
from typing import TypedDict


class PickPoint(TypedDict):
    station: str
    ts_p: float
    s_minus_p: float
    rejected: bool
    cx: float
    cy: float


class Tick(TypedDict):
    label: str
    pos: float


# --- plot geometry (SVG user units) ---
_W, _H = 600.0, 380.0
_LEFT, _RIGHT, _TOP, _BOTTOM = 62.0, 574.0, 26.0, 328.0
_XMAX, _YMAX = 21.0, 16.0


def _px(x: float) -> float:
    return _LEFT + (x / _XMAX) * (_RIGHT - _LEFT)


def _py(y: float) -> float:
    return _BOTTOM - (y / _YMAX) * (_BOTTOM - _TOP)


# Sample single-event pick table: station, t_P, t_S - t_P, rejected
_RAW: list[tuple[str, float, float, bool]] = [
    ("BRG", 2.6, 1.94, False),
    ("MOX", 4.1, 2.95, False),
    ("CLL", 5.5, 4.06, False),
    ("TANN", 6.9, 5.01, False),
    ("WERD", 8.2, 7.88, True),
    ("PLN", 9.6, 7.05, False),
    ("SCHF", 11.1, 8.05, False),
    ("GUNZ", 12.4, 9.11, False),
    ("ROHR", 14.0, 10.18, False),
    ("NEUB", 15.8, 9.94, True),
    ("LAUE", 17.2, 12.61, False),
    ("HAIN", 18.9, 13.75, False),
]

_POINTS: list[PickPoint] = [
    PickPoint(
        station=s,
        ts_p=x,
        s_minus_p=y,
        rejected=r,
        cx=round(_px(x), 2),
        cy=round(_py(y), 2),
    )
    for s, x, y, r in _RAW
]


def _fit(
    rows: list[tuple[str, float, float, bool]],
) -> tuple[float, float, float]:
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / ((sxx * syy) ** 0.5)
    return slope, intercept, r


_kept = [r for r in _RAW if not r[3]]
_SLOPE, _INT, _R = _fit(_kept)


class WadatiState(rx.State):
    """Static sample event used by the launch page's Wadati centerpiece."""

    event_id: str = "2024-05-17T04:12:09Z"
    points: list[PickPoint] = _POINTS
    x_ticks: list[Tick] = [
        Tick(label=str(v), pos=round(_px(float(v)), 2))
        for v in (0, 5, 10, 15, 20)
    ]
    y_ticks: list[Tick] = [
        Tick(label=str(v), pos=round(_py(float(v)), 2))
        for v in (0, 4, 8, 12, 16)
    ]

    plot_w: float = _W
    plot_h: float = _H
    axis_left: float = _LEFT
    axis_right: float = _RIGHT
    axis_top: float = _TOP
    axis_bottom: float = _BOTTOM

    fit_x1: float = round(_px(0.0), 2)
    fit_y1: float = round(_py(_INT), 2)
    fit_x2: float = round(_px(_XMAX), 2)
    fit_y2: float = round(_py(_INT + _SLOPE * _XMAX), 2)

    vp_vs: float = round(_SLOPE + 1.0, 3)
    correlation: float = round(_R, 4)
    retained: int = len(_kept)
    total: int = len(_RAW)

    @rx.var
    def rejected_stations(self) -> str:
        names = [r[0] for r in _RAW if r[3]]
        return ", ".join(names) if names else "none"

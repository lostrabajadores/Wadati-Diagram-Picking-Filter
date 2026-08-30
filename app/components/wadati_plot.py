import reflex as rx

from app.states.wadati_state import PickPoint, Tick, WadatiState

TEAL = "#0E6B6B"
RED = "#A6321F"
RULE = "#CBBFA6"
INK = "#2B2F33"
SLATE = "#5C666F"


def _x_tick(tick: Tick) -> rx.Component:
    return rx.fragment(
        rx.el.svg.line(
            x1=tick["pos"].to_string(),
            x2=tick["pos"].to_string(),
            y1=WadatiState.axis_top.to_string(),
            y2=WadatiState.axis_bottom.to_string(),
            stroke=RULE,
            stroke_width=0.5,
            stroke_dasharray="2 4",
        ),
        rx.el.svg.text(
            tick["label"],
            x=tick["pos"].to_string(),
            y=(WadatiState.axis_bottom + 18).to_string(),
            fill=SLATE,
            font_size="11",
            text_anchor="middle",
        ),
    )


def _y_tick(tick: Tick) -> rx.Component:
    return rx.fragment(
        rx.el.svg.line(
            x1=WadatiState.axis_left.to_string(),
            x2=WadatiState.axis_right.to_string(),
            y1=tick["pos"].to_string(),
            y2=tick["pos"].to_string(),
            stroke=RULE,
            stroke_width=0.5,
            stroke_dasharray="2 4",
        ),
        rx.el.svg.text(
            tick["label"],
            x=(WadatiState.axis_left - 10).to_string(),
            y=(tick["pos"] + 4).to_string(),
            fill=SLATE,
            font_size="11",
            text_anchor="end",
        ),
    )


def _pick(point: PickPoint) -> rx.Component:
    return rx.fragment(
        rx.cond(
            point["rejected"],
            rx.el.svg.path(
                d="M -5 -5 L 5 5 M 5 -5 L -5 5",
                transform=f"translate({point['cx']} {point['cy']})",
                stroke=RED,
                stroke_width=2.2,
                stroke_linecap="round",
            ),
            rx.el.svg.circle(
                cx=point["cx"].to_string(),
                cy=point["cy"].to_string(),
                r=5,
                fill=TEAL,
                stroke="#F6F1E7",
                stroke_width=1.2,
            ),
        ),
        rx.el.svg.text(
            point["station"],
            x=(point["cx"] + 8).to_string(),
            y=(point["cy"] - 7).to_string(),
            fill=rx.cond(point["rejected"], RED, SLATE),
            font_size="9.5",
            letter_spacing="0.04em",
        ),
    )


def wadati_plot() -> rx.Component:
    return rx.el.div(
        rx.el.svg(
            rx.foreach(WadatiState.y_ticks, _y_tick),
            rx.foreach(WadatiState.x_ticks, _x_tick),
            rx.el.svg.line(
                x1=WadatiState.axis_left.to_string(),
                x2=WadatiState.axis_right.to_string(),
                y1=WadatiState.axis_bottom.to_string(),
                y2=WadatiState.axis_bottom.to_string(),
                stroke=INK,
                stroke_width=1,
            ),
            rx.el.svg.line(
                x1=WadatiState.axis_left.to_string(),
                x2=WadatiState.axis_left.to_string(),
                y1=WadatiState.axis_top.to_string(),
                y2=WadatiState.axis_bottom.to_string(),
                stroke=INK,
                stroke_width=1,
            ),
            rx.el.svg.line(
                x1=WadatiState.fit_x1.to_string(),
                x2=WadatiState.fit_x2.to_string(),
                y1=WadatiState.fit_y1.to_string(),
                y2=WadatiState.fit_y2.to_string(),
                stroke=TEAL,
                stroke_width=1.6,
            ),
            rx.foreach(WadatiState.points, _pick),
            rx.el.svg.text(
                "P travel time  t_P  (s)",
                x=(
                    (WadatiState.axis_left + WadatiState.axis_right) / 2
                ).to_string(),
                y=(WadatiState.axis_bottom + 44).to_string(),
                fill=INK,
                font_size="11.5",
                text_anchor="middle",
                letter_spacing="0.06em",
            ),
            rx.el.svg.text(
                "t_S − t_P  (s)",
                x="18",
                y=(
                    (WadatiState.axis_top + WadatiState.axis_bottom) / 2
                ).to_string(),
                fill=INK,
                font_size="11.5",
                text_anchor="middle",
                transform="rotate(-90 18 177)",
                letter_spacing="0.06em",
            ),
            view_box=f"0 0 {600} {380}",
            class_name="w-full h-auto",
        ),
        class_name="w-full",
    )

import reflex as rx

from app.states.setup_state import SetupState, Step


def _step(step: Step) -> rx.Component:
    return rx.el.li(
        rx.el.div(
            rx.el.span(
                step["index"],
                class_name="text-[10px] tracking-[0.28em] text-[#8a7f68] font-semibold pt-1",
            ),
            rx.el.div(
                rx.el.h3(
                    step["title"],
                    class_name="text-[15px] font-semibold text-[#2B2F33] tracking-tight",
                ),
                rx.el.p(
                    step["detail"],
                    class_name="text-[13px] text-[#5C666F] leading-relaxed mt-1",
                ),
                rx.el.div(
                    rx.el.code(
                        step["command"],
                        class_name="text-[12px] text-[#0E6B6B] font-mono break-all",
                    ),
                    rx.el.button(
                        rx.cond(
                            SetupState.copied == step["command"],
                            rx.icon("check", class_name="h-3.5 w-3.5"),
                            rx.icon("copy", class_name="h-3.5 w-3.5"),
                        ),
                        on_click=lambda: SetupState.copy(step["command"]),
                        class_name="shrink-0 text-[#8a7f68] hover:text-[#0E6B6B] transition-colors cursor-pointer",
                        title="Copy command",
                    ),
                    class_name="mt-3 flex items-start justify-between gap-3 border border-[#CBBFA6] bg-[#EDE5D6]/60 px-3 py-2",
                ),
                class_name="flex-1 min-w-0",
            ),
            class_name="flex gap-5",
        ),
        class_name="py-5 border-t border-[#CBBFA6]/70 first:border-t-0",
    )


def setup_steps() -> rx.Component:
    return rx.el.ol(
        rx.foreach(SetupState.steps, _step),
        class_name="w-full flex flex-col",
    )

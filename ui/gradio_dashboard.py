#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.studio_backend import (  # noqa: E402
    MODEL_PROFILE_CHOICES,
    get_pipeline_runtime_snapshot,
    get_service_status,
    start_pipeline_run,
    stop_pipeline_run,
)

RUN_MODE_CHOICES = ["Ask Every Run", "One Chapter"]
CHAPTER_ACTION_CHOICES = ["Prompt each time", "Rebuild", "Skip", "Cancel"]


def _start_run(
    project_name: str,
    run_mode: str,
    start_chapter: float,
    last_chapter: float,
    target_chapter: float,
    existing_action: str,
    model_profile: str,
) -> tuple[str, str, str, str, str]:
    message = start_pipeline_run(
        project_name=project_name.strip() or "first_story",
        run_mode=run_mode,
        start_chapter=int(start_chapter),
        last_chapter=int(last_chapter),
        target_chapter=int(target_chapter),
        existing_chapter_action=existing_action,
        model_profile=model_profile,
    )
    status, files, packet, log_tail = get_pipeline_runtime_snapshot(int(start_chapter), int(last_chapter))
    return message, status, files, packet, log_tail


def _stop_run(project_name: str, start_chapter: float, last_chapter: float) -> tuple[str, str, str, str, str]:
    message = stop_pipeline_run(project_name.strip() or "first_story")
    status, files, packet, log_tail = get_pipeline_runtime_snapshot(int(start_chapter), int(last_chapter))
    return message, status, files, packet, log_tail


def _refresh(start_chapter: float, last_chapter: float) -> tuple[str, str, str, str]:
    return get_pipeline_runtime_snapshot(int(start_chapter), int(last_chapter))


def _service_status() -> str:
    return get_service_status()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Story Engine Dashboard") as demo:
        gr.Markdown("# Story Engine Dashboard")
        gr.Markdown(
            "This dashboard provides active/current-work visibility for sections 5.2-9: "
            "run control, chapter progress, current phase, review packet path, and log tail."
        )

        with gr.Row():
            project_name = gr.Textbox(label="Project Name", value="first_story")
            run_mode = gr.Dropdown(label="Run Mode", choices=RUN_MODE_CHOICES, value="Ask Every Run")
            model_profile = gr.Dropdown(
                label="Model Profile",
                choices=MODEL_PROFILE_CHOICES,
                value=MODEL_PROFILE_CHOICES[-1],
            )

        with gr.Row():
            start_chapter = gr.Number(label="Start Chapter", value=1, precision=0)
            last_chapter = gr.Number(label="Last Chapter", value=3, precision=0)
            target_chapter = gr.Number(label="Target Chapter (One Chapter mode)", value=1, precision=0)

        existing_action = gr.Dropdown(
            label="If Target Chapter Already Exists",
            choices=CHAPTER_ACTION_CHOICES,
            value="Prompt each time",
        )

        with gr.Row():
            start_button = gr.Button("Start Run", variant="primary")
            stop_button = gr.Button("Stop Run", variant="stop")
            refresh_button = gr.Button("Refresh Status")
            service_button = gr.Button("Refresh Services")

        action_message = gr.Textbox(label="Action Message", lines=2)
        service_status = gr.Textbox(label="Service Status", lines=8)

        with gr.Row():
            status_box = gr.Textbox(label="Active Status", lines=12)
            files_box = gr.Textbox(label="Current Chapter Artifacts", lines=12)

        with gr.Row():
            packet_box = gr.Textbox(label="Review Packet", lines=3)
            log_box = gr.Textbox(label="Run Log Tail", lines=12)

        start_button.click(
            _start_run,
            inputs=[project_name, run_mode, start_chapter, last_chapter, target_chapter, existing_action, model_profile],
            outputs=[action_message, status_box, files_box, packet_box, log_box],
        )
        stop_button.click(
            _stop_run,
            inputs=[project_name, start_chapter, last_chapter],
            outputs=[action_message, status_box, files_box, packet_box, log_box],
        )
        refresh_button.click(
            _refresh,
            inputs=[start_chapter, last_chapter],
            outputs=[status_box, files_box, packet_box, log_box],
        )
        service_button.click(_service_status, outputs=[service_status])

        demo.load(_service_status, outputs=[service_status])
        demo.load(
            _refresh,
            inputs=[start_chapter, last_chapter],
            outputs=[status_box, files_box, packet_box, log_box],
            every=5,
        )

    return demo


def main() -> None:
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7800, share=False)


if __name__ == "__main__":
    main()

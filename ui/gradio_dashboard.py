#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
import time

import gradio as gr

from ui.studio_backend import (
    MODEL_PROFILE_CHOICES,
    MODEL_PROFILE_QWEN25_Q5,
    approve_review_marker,
    clear_project_data,
    clear_run_logs,
    create_guide_template,
    create_project,
    get_default_chapter_range,
    get_pipeline_runtime_snapshot,
    get_required_input_windows,
    get_service_status,
    get_download_path,
    get_project_voice_download_path,
    get_readiness_report,
    import_uploaded_file,
    import_project_voice,
    import_last_signal_sources,
    list_downloadable_files,
    list_project_voices,
    load_input_text,
    load_json_preview,
    load_narration_text,
    project_overview,
    refresh_projects,
    run_conversion,
    save_input_text,
    save_narration_text,
    select_project,
    switch_project_cleanup,
    reset_pipeline_run,
    start_pipeline_run,
    stop_pipeline_run,
    sync_selected_voice_to_root,
    sync_project_json_to_root,
)

GENRE_WORD_PRESETS = {
    "Auto (Current Settings)": (1800, 2400),
    "Thriller": (2400, 3200),
    "Sci-Fi": (2600, 3600),
    "Fantasy": (2800, 3800),
    "Mystery": (2200, 3000),
    "Horror": (1800, 2600),
    "Romance": (2200, 3000),
    "Literary": (2600, 3400),
}

INPUT_CHOICES = [
    ("Story DNA Summary", "dna"),
    ("Story Bible", "bible"),
    ("Chapter Blueprint", "blueprint"),
    ("Style Guide / Phase 4 Writing Prompts", "style_guide"),
    ("Consistency Checklist", "consistency"),
]

JSON_CHOICES = [
    "story_bible.json",
    "characters.json",
    "chapter_briefs.json",
    "story_engine_conversion_prompt.md",
]

TIP_JS = """
() => {
    if (window.__storyStudioTipsBound) return;
    window.__storyStudioTipsBound = true;
    const bubble = document.createElement("div");
    bubble.id = "story-studio-tip-bubble";
    bubble.style.display = "none";
    document.body.appendChild(bubble);

    let hideTimer = null;
    const hideNow = () => {
        bubble.style.display = "none";
        bubble.textContent = "";
    };

    const show = (anchor) => {
        const msg = anchor?.dataset?.tip || "";
        if (!msg) return;
        bubble.textContent = msg;
        const rect = anchor.getBoundingClientRect();
        bubble.style.left = `${rect.left + window.scrollX}px`;
        bubble.style.top = `${rect.bottom + window.scrollY + 8}px`;
        bubble.style.display = "block";
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(hideNow, 2000);
    };

    document.addEventListener("mouseenter", (ev) => {
        const anchor = ev.target.closest(".ss-tip-anchor");
        if (anchor) show(anchor);
    }, true);

    document.addEventListener("focusin", (ev) => {
        const anchor = ev.target.closest(".ss-tip-anchor");
        if (anchor) show(anchor);
    });

    document.addEventListener("mouseleave", (ev) => {
        const anchor = ev.target.closest(".ss-tip-anchor");
        if (anchor) hideNow();
    }, true);
}
"""

TIP_CSS = """
.ss-tip-anchor {
    display: inline-flex;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    border: 1px solid #9aa4b2;
    color: #415169;
    font-size: 12px;
    line-height: 1;
    align-items: center;
    justify-content: center;
    cursor: help;
    user-select: none;
}

#story-studio-tip-bubble {
    position: absolute;
    z-index: 9999;
    max-width: 360px;
    background: #111827;
    color: #f9fafb;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
}
"""


def _tip(text: str) -> str:
    safe_text = text.replace('"', "&quot;")
    return f'<span class="ss-tip-anchor" tabindex="0" data-tip="{safe_text}" aria-label="Help">?</span>'


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred_port: int, max_port: int, strict_port: bool) -> int:
    if strict_port:
        if _port_available(host, preferred_port):
            return preferred_port
        raise RuntimeError(
            f"Story Studio strict-port mode is enabled and port {preferred_port} is unavailable. "
            "Stop the process using that port or set STUDIO_STRICT_PORT=0."
        )

    stop_port = max(preferred_port, max_port)
    for candidate in range(preferred_port, stop_port + 1):
        if _port_available(host, candidate):
            return candidate

    raise RuntimeError(
        f"No open port found in range {preferred_port}-{stop_port}. "
        "Set STUDIO_PORT/STUDIO_PORT_MAX or free an existing listener."
    )


def build_app() -> gr.Blocks:
    projects, active_project = refresh_projects()
    default_start_chapter, default_last_chapter = get_default_chapter_range()

    with gr.Blocks(title="Story Studio") as demo:
        gr.Markdown("# Story Studio (Local v1)\nProject-first workflow with path-free conversion inputs.")

        with gr.Row():
            project_dropdown = gr.Dropdown(
                choices=projects,
                value=active_project if active_project else None,
                label="Active Project",
                allow_custom_value=False,
            )
            gr.HTML(_tip("Set one active project at a time. All saves, conversion, and sync actions use this project."))
            refresh_btn = gr.Button("Refresh Projects")
            activate_btn = gr.Button("Set Active")

        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Projects"):
            project_name = gr.Textbox(label="Create New Project", placeholder="example: the-gap-protocol")
            create_btn = gr.Button("Create Project")
            with gr.Row():
                switch_cleanup_btn = gr.Button("Switch Cleanup (Clear Root Runtime)")
                switch_cleanup_force_stop = gr.Checkbox(label="Force Stop Active Pipeline", value=False)
            switch_cleanup_status = gr.Textbox(label="Switch Cleanup Status", interactive=False)
            with gr.Row():
                start_fresh_btn = gr.Button("Start Fresh (Clear Project Inputs + JSON)")
                start_fresh_force_stop = gr.Checkbox(label="Force Stop Active Pipeline", value=False)
                start_fresh_clear_root = gr.Checkbox(label="Also Clear Root Pipeline Files", value=True)
            start_fresh_status = gr.Textbox(label="Start Fresh Status", interactive=False)
            project_summary = gr.Textbox(label="Project Overview", lines=12, interactive=False)

        with gr.Tab("Inputs"):
            gr.Markdown(
                "Workflow: upload/paste source docs, then run conversion. "
                "Conversion now generates style_guide.txt, consistency_checklist.txt, and master_system_prompt.md automatically. "
                "You can still override guide files here using templates or manual edits."
            )
            input_key = gr.Dropdown(choices=INPUT_CHOICES, value="dna", label="Input Document")
            gr.HTML(_tip("Use Style Guide and Consistency Checklist for quality control. You can create starter templates below."))
            input_text = gr.Textbox(label="Document Text", lines=20)
            with gr.Row():
                load_input_btn = gr.Button("Load")
                save_input_btn = gr.Button("Save")
            with gr.Row():
                style_template_btn = gr.Button("Create Style Guide Template")
                consistency_template_btn = gr.Button("Create Consistency Template")
            input_upload = gr.File(label="Upload .txt to selected input", file_count="single", type="filepath")
            upload_btn = gr.Button("Import Uploaded File")
            import_last_signal_btn = gr.Button("Import The Last Signal Sources")
            input_status = gr.Textbox(label="Input Status", interactive=False)

            gr.Markdown("### Required Source Windows (Fixed Slots)")
            gr.Markdown("Use these windows to verify each required source slot before conversion.")
            with gr.Row():
                refresh_required_windows_btn = gr.Button("Refresh Required Windows")
            required_windows_status = gr.Textbox(label="Required Window Status", interactive=False)
            dna_window = gr.Textbox(label="Story DNA Summary Slot", lines=8, interactive=False)
            bible_window = gr.Textbox(label="Story Bible Slot", lines=8, interactive=False)
            blueprint_window = gr.Textbox(label="Chapter Blueprint Slot", lines=8, interactive=False)
            style_window = gr.Textbox(label="Style Guide / Phase 4 Writing Prompts Slot", lines=8, interactive=False)

        with gr.Tab("Convert"):
            mode = gr.Radio(choices=["rule", "prompt", "hybrid"], value="rule", label="Conversion Mode")
            gr.HTML(_tip("Conversion is locked until Story DNA, Story Bible, Chapter Blueprint, and Style Guide (or Phase 4 Writing Prompts) are all present and non-empty."))
            convert_btn = gr.Button("Run Conversion")
            convert_log = gr.Textbox(label="Conversion Log", lines=10, interactive=False)

            readiness_btn = gr.Button("Run Readiness Check")
            readiness_report = gr.Textbox(label="Readiness Report", lines=10, interactive=False)

            with gr.Row():
                json_file = gr.Dropdown(choices=JSON_CHOICES, value="story_bible.json", label="Preview Output")
                preview_btn = gr.Button("Load Preview")
            preview_text = gr.Textbox(label="Output Preview", lines=20, interactive=False)

            sync_btn = gr.Button("Sync Project JSON + Guides to Root Pipeline")
            sync_status = gr.Textbox(label="Sync Status", interactive=False)

        with gr.Tab("Voice"):
            gr.Markdown(
                "Upload a user-downloaded narrator voice file for this project, then sync selected voice to root `.env` "
                "(`VOICE_SAMPLE`) for pipeline use."
            )
            gr.Markdown("Accepted by Chatterbox workflow: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` (`.wav` recommended).")
            with gr.Row():
                voice_upload = gr.File(label="Upload Voice File", file_count="single", type="filepath")
                upload_voice_btn = gr.Button("Import Voice")
            with gr.Row():
                voice_choice = gr.Dropdown(choices=[], label="Project Voices")
                refresh_voices_btn = gr.Button("Refresh Voices")
            with gr.Row():
                sync_voice_btn = gr.Button("Sync Selected Voice to Root + .env")
                prepare_voice_download_btn = gr.Button("Prepare Voice Download")
            voice_download = gr.File(label="Voice Download", interactive=False)
            voice_status = gr.Textbox(label="Voice Status", interactive=False)

        with gr.Tab("Downloads"):
            reload_downloads_btn = gr.Button("Refresh File List")
            download_choice = gr.Dropdown(choices=[], label="Project File")
            choose_download_btn = gr.Button("Prepare Download")
            download_file = gr.File(label="Download", interactive=False)

        with gr.Tab("Run Dashboard"):
            gr.Markdown("Unified run controls: start services checks, launch pipeline mode, monitor progress/phase, approve review pauses, and edit narration text.")

            with gr.Row():
                run_mode = gr.Radio(
                    choices=["One Chapter", "Sequential", "Resume"],
                    value="Sequential",
                    label="Run Mode",
                )
                model_profile = gr.Dropdown(
                    choices=MODEL_PROFILE_CHOICES,
                    value=MODEL_PROFILE_QWEN25_Q5,
                    label="LLM Model Profile",
                )
                start_chapter = gr.Number(value=default_start_chapter, precision=0, label="Start Chapter")
                last_chapter = gr.Number(value=default_last_chapter, precision=0, label="Last Chapter")
                chapter_complete_alert = gr.Dropdown(
                    choices=["Double Beep", "Gong", "Off"],
                    value="Double Beep",
                    label="Chapter Complete Alert",
                )
                one_chapter_target = gr.Number(value=1, precision=0, label="One Chapter Target", visible=False)
                existing_chapter_action = gr.Dropdown(
                    choices=["Prompt each time", "Rebuild", "Skip", "Cancel"],
                    value="Prompt each time",
                    label="If Target Already Exists",
                    visible=False,
                )
            with gr.Row():
                genre_preset = gr.Dropdown(
                    choices=list(GENRE_WORD_PRESETS.keys()),
                    value="Auto (Current Settings)",
                    label="Genre Word Target Preset",
                )
                word_target_min = gr.Slider(minimum=800, maximum=6000, step=100, value=1800, label="Word Target Min")
                word_target_max = gr.Slider(minimum=1200, maximum=7000, step=100, value=2400, label="Word Target Max")
                narration_speed = gr.Slider(minimum=0.7, maximum=1.3, step=0.01, value=1.0, label="Narration Pace")
            with gr.Row():
                start_run_btn = gr.Button("Start Pipeline")
                stop_run_btn = gr.Button("Stop Pipeline")
                refresh_dashboard_btn = gr.Button("Refresh Dashboard")
                clear_run_logs_btn = gr.Button("Clear Run Logs")
            with gr.Row():
                auto_refresh_enabled = gr.Checkbox(label="Auto Refresh", value=False)
                auto_refresh_seconds = gr.Slider(minimum=30, maximum=60, step=5, value=45, label="Auto Refresh Seconds")
                timer_supported = hasattr(gr, "Timer")
                auto_refresh_default_status = (
                    "Timer support detected. Auto refresh is disabled until enabled."
                    if timer_supported
                    else "Timer not supported by this Gradio build; auto refresh unavailable."
                )
                auto_refresh_status = gr.Textbox(
                    label="Auto Refresh Status",
                    value=auto_refresh_default_status,
                    interactive=False,
                )
            with gr.Row():
                reset_run_btn = gr.Button("Reset Run")
                reset_scope = gr.Radio(
                    choices=["Current Chapter", "All Chapters", "Runner State Only"],
                    value="Current Chapter",
                    label="Reset Scope",
                )
                reset_chapter = gr.Number(value=1, precision=0, label="Reset Chapter")
            with gr.Row():
                reset_force_stop = gr.Checkbox(label="Force Stop Active Pipeline", value=False)
                reset_confirm_all = gr.Checkbox(label="Confirm All Chapters Reset", value=False)
            dashboard_status = gr.Textbox(label="Dashboard Status", interactive=False)
            service_status = gr.Textbox(label="Service Status", lines=6, interactive=False)
            phase_status = gr.Textbox(label="Run Snapshot", lines=8, interactive=False)
            artifact_status = gr.Textbox(label="Current Chapter Artifacts", lines=10, interactive=False)
            review_packet = gr.Textbox(label="Review Packet", interactive=False)
            run_log = gr.Textbox(label="Run Log (tail)", lines=16, interactive=False)

            with gr.Row():
                review_chapter = gr.Number(value=1, precision=0, label="Review Chapter")
                review_stage = gr.Radio(
                    choices=["pre_narration", "post_chapter"],
                    value="pre_narration",
                    label="Approve Stage",
                )
                approve_btn = gr.Button("Create Approval Marker")
            review_status = gr.Textbox(label="Review Action Status", interactive=False)

            with gr.Row():
                narration_chapter = gr.Number(value=1, precision=0, label="Narration Chapter")
                load_narration_btn = gr.Button("Load Narration Text")
                save_narration_btn = gr.Button("Save Narration Text")
            narration_text = gr.Textbox(label="Narration Text (chXX_tts.txt)", lines=14)
            narration_status = gr.Textbox(label="Narration Status", interactive=False)

            refresh_timer = gr.Timer(value=45, active=False) if timer_supported else None

        def _refresh_state() -> tuple[gr.Dropdown, str, str]:
            p, active = refresh_projects()
            return gr.Dropdown(choices=p, value=active if active else None), f"Active project: {active or 'none'}", project_overview(active)

        def _set_active(selected: str) -> tuple[str, str]:
            msg = select_project(selected)
            return msg, project_overview(selected)

        def _create_project(name: str) -> tuple[gr.Dropdown, str, str]:
            choices, active, message = create_project(name)
            return gr.Dropdown(choices=choices, value=active), message, project_overview(active)

        def _load_input(project: str, key: str) -> str:
            return load_input_text(project, key)

        def _save_input(project: str, key: str, text: str) -> str:
            return save_input_text(project, key, text)

        def _import_file(project: str, key: str, uploaded: str | None) -> str:
            return import_uploaded_file(project, key, uploaded)

        def _import_last_signal(project: str) -> str:
            return import_last_signal_sources(project)

        def _create_template(project: str, key: str) -> tuple[str, str, str]:
            status, text = create_guide_template(project, key)
            return key, text, status

        def _create_style_template(project: str) -> tuple[str, str, str]:
            return _create_template(project, "style_guide")

        def _create_consistency_template(project: str) -> tuple[str, str, str]:
            return _create_template(project, "consistency")

        def _convert(project: str, selected_mode: str) -> str:
            return run_conversion(project, selected_mode)

        def _readiness(project: str) -> str:
            return get_readiness_report(project)

        def _preview(project: str, selected_json: str) -> str:
            return load_json_preview(project, selected_json)

        def _sync(project: str) -> str:
            return sync_project_json_to_root(project)

        def _refresh_required_windows(project: str) -> tuple[str, str, str, str, str]:
            return get_required_input_windows(project)

        def _start_fresh(project: str, force_stop: bool, clear_root: bool) -> str:
            return clear_project_data(project, bool(force_stop), bool(clear_root))

        def _reload_downloads(project: str) -> gr.Dropdown:
            files = list_downloadable_files(project)
            value = files[0] if files else None
            return gr.Dropdown(choices=files, value=value)

        def _prepare_download(project: str, selected_file: str) -> str | None:
            return get_download_path(project, selected_file)

        def _reload_voices(project: str) -> gr.Dropdown:
            voices = list_project_voices(project)
            value = voices[0] if voices else None
            return gr.Dropdown(choices=voices, value=value)

        def _import_voice(project: str, uploaded: str | None) -> tuple[str, gr.Dropdown]:
            msg, voices, selected = import_project_voice(project, uploaded)
            return msg, gr.Dropdown(choices=voices, value=selected)

        def _sync_voice(project: str, selected_voice: str) -> str:
            return sync_selected_voice_to_root(project, selected_voice)

        def _prepare_voice_download(project: str, selected_voice: str) -> str | None:
            return get_project_voice_download_path(project, selected_voice)

        def _start_run(
            project: str,
            mode_name: str,
            model_choice: str,
            start_num: float,
            last_num: float,
            min_words: float,
            max_words: float,
            pace: float,
            target_chapter: float,
            chapter_exists_action: str,
            alert_mode: str,
        ) -> str:
            min_i = int(min_words or 0)
            max_i = int(max_words or 0)
            if min_i > 0 and max_i > 0 and min_i > max_i:
                min_i, max_i = max_i, min_i
            target_i = int(target_chapter or 0)
            return start_pipeline_run(
                project,
                mode_name,
                int(start_num or 0),
                int(last_num or 0),
                min_i,
                max_i,
                float(pace or 1.0),
                target_i,
                chapter_exists_action or "Prompt each time",
                model_choice,
                alert_mode,
            )

        def _run_mode_ui(mode_name: str):
            one_mode = (mode_name or "").strip().lower().startswith("one")
            return (
                gr.update(visible=one_mode),
                gr.update(visible=one_mode),
            )

        def _apply_genre_preset(name: str) -> tuple[float, float]:
            min_words, max_words = GENRE_WORD_PRESETS.get(name, GENRE_WORD_PRESETS["Auto (Current Settings)"])
            return float(min_words), float(max_words)

        def _stop_run(project: str) -> str:
            return stop_pipeline_run(project)

        def _switch_cleanup(project: str, force_stop: bool) -> str:
            return switch_project_cleanup(project or "", bool(force_stop))

        def _reset_run(
            project: str,
            scope: str,
            chapter_num: float,
            force_stop: bool,
            confirm_all: bool,
        ) -> str:
            return reset_pipeline_run(
                project or "",
                scope,
                int(chapter_num or 1),
                bool(force_stop),
                bool(confirm_all),
            )

        def _refresh_dashboard(start_num: float, last_num: float) -> tuple[str, str, str, str]:
            return get_pipeline_runtime_snapshot(int(start_num or 0), int(last_num or 0))

        def _clear_logs(start_num: float, last_num: float) -> tuple[str, str, str, str, str]:
            msg = clear_run_logs()
            phase, artifacts, packet, log = _refresh_dashboard(start_num, last_num)
            return msg, phase, artifacts, packet, log

        def _auto_refresh_toggle(enabled: bool, seconds: float) -> tuple[gr.Timer, str]:
            sec = int(seconds or 45)
            sec = max(30, min(60, sec))
            status = f"Auto refresh {'enabled' if enabled else 'disabled'} ({sec}s)."
            return gr.Timer(value=sec, active=bool(enabled)), status

        def _auto_refresh_pulse(start_num: float, last_num: float) -> tuple[str, str, str, str]:
            return _refresh_dashboard(start_num, last_num)

        def _auto_refresh_service() -> str:
            stamp = time.strftime("%H:%M:%S")
            return get_service_status() + f"\n\nLast refresh: {stamp}"

        def _service_status() -> str:
            return get_service_status()

        def _approve(chapter_num: float, stage: str) -> str:
            return approve_review_marker(int(chapter_num or 1), stage)

        def _load_narration(chapter_num: float) -> str:
            return load_narration_text(int(chapter_num or 1))

        def _save_narration(chapter_num: float, text: str) -> str:
            return save_narration_text(int(chapter_num or 1), text)

        refresh_btn.click(_refresh_state, inputs=[], outputs=[project_dropdown, status_box, project_summary])
        activate_btn.click(_set_active, inputs=[project_dropdown], outputs=[status_box, project_summary])

        create_btn.click(_create_project, inputs=[project_name], outputs=[project_dropdown, status_box, project_summary])
        switch_cleanup_btn.click(
            _switch_cleanup,
            inputs=[project_dropdown, switch_cleanup_force_stop],
            outputs=[switch_cleanup_status],
        )
        start_fresh_btn.click(
            _start_fresh,
            inputs=[project_dropdown, start_fresh_force_stop, start_fresh_clear_root],
            outputs=[start_fresh_status],
        )

        load_input_btn.click(_load_input, inputs=[project_dropdown, input_key], outputs=[input_text])
        save_input_btn.click(_save_input, inputs=[project_dropdown, input_key, input_text], outputs=[input_status])
        upload_btn.click(_import_file, inputs=[project_dropdown, input_key, input_upload], outputs=[input_status])
        import_last_signal_btn.click(_import_last_signal, inputs=[project_dropdown], outputs=[input_status])
        style_template_btn.click(_create_style_template, inputs=[project_dropdown], outputs=[input_key, input_text, input_status])
        consistency_template_btn.click(
            _create_consistency_template,
            inputs=[project_dropdown],
            outputs=[input_key, input_text, input_status],
        )
        refresh_required_windows_btn.click(
            _refresh_required_windows,
            inputs=[project_dropdown],
            outputs=[dna_window, bible_window, blueprint_window, style_window, required_windows_status],
        )

        convert_btn.click(_convert, inputs=[project_dropdown, mode], outputs=[convert_log])
        readiness_btn.click(_readiness, inputs=[project_dropdown], outputs=[readiness_report])
        preview_btn.click(_preview, inputs=[project_dropdown, json_file], outputs=[preview_text])
        sync_btn.click(_sync, inputs=[project_dropdown], outputs=[sync_status])

        refresh_voices_btn.click(_reload_voices, inputs=[project_dropdown], outputs=[voice_choice])
        upload_voice_btn.click(_import_voice, inputs=[project_dropdown, voice_upload], outputs=[voice_status, voice_choice])
        sync_voice_btn.click(_sync_voice, inputs=[project_dropdown, voice_choice], outputs=[voice_status])
        prepare_voice_download_btn.click(
            _prepare_voice_download,
            inputs=[project_dropdown, voice_choice],
            outputs=[voice_download],
        )

        reload_downloads_btn.click(_reload_downloads, inputs=[project_dropdown], outputs=[download_choice])
        choose_download_btn.click(_prepare_download, inputs=[project_dropdown, download_choice], outputs=[download_file])

        genre_preset.change(_apply_genre_preset, inputs=[genre_preset], outputs=[word_target_min, word_target_max])
        run_mode.change(_run_mode_ui, inputs=[run_mode], outputs=[one_chapter_target, existing_chapter_action])

        start_run_btn.click(
            _start_run,
            inputs=[
                project_dropdown,
                run_mode,
                model_profile,
                start_chapter,
                last_chapter,
                word_target_min,
                word_target_max,
                narration_speed,
                one_chapter_target,
                existing_chapter_action,
                chapter_complete_alert,
            ],
            outputs=[dashboard_status],
        )
        stop_run_btn.click(_stop_run, inputs=[project_dropdown], outputs=[dashboard_status])
        reset_run_btn.click(
            _reset_run,
            inputs=[project_dropdown, reset_scope, reset_chapter, reset_force_stop, reset_confirm_all],
            outputs=[dashboard_status],
        )
        refresh_dashboard_btn.click(
            _refresh_dashboard,
            inputs=[start_chapter, last_chapter],
            outputs=[phase_status, artifact_status, review_packet, run_log],
        )
        refresh_dashboard_btn.click(_service_status, inputs=[], outputs=[service_status])
        clear_run_logs_btn.click(
            _clear_logs,
            inputs=[start_chapter, last_chapter],
            outputs=[dashboard_status, phase_status, artifact_status, review_packet, run_log],
        )

        if refresh_timer is not None:
            auto_refresh_enabled.change(
                _auto_refresh_toggle,
                inputs=[auto_refresh_enabled, auto_refresh_seconds],
                outputs=[refresh_timer, auto_refresh_status],
            )
            auto_refresh_seconds.change(
                _auto_refresh_toggle,
                inputs=[auto_refresh_enabled, auto_refresh_seconds],
                outputs=[refresh_timer, auto_refresh_status],
            )
            refresh_timer.tick(
                _auto_refresh_pulse,
                inputs=[start_chapter, last_chapter],
                outputs=[phase_status, artifact_status, review_packet, run_log],
            )
            refresh_timer.tick(_auto_refresh_service, inputs=[], outputs=[service_status])

        approve_btn.click(_approve, inputs=[review_chapter, review_stage], outputs=[review_status])
        load_narration_btn.click(_load_narration, inputs=[narration_chapter], outputs=[narration_text])
        save_narration_btn.click(
            _save_narration,
            inputs=[narration_chapter, narration_text],
            outputs=[narration_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    host = os.getenv("STUDIO_HOST", "127.0.0.1")
    preferred_port = _env_int("STUDIO_PORT", 7800)
    max_port = _env_int("STUDIO_PORT_MAX", 7810)
    strict_port = _env_bool("STUDIO_STRICT_PORT", False)

    try:
        port = _pick_port(host, preferred_port, max_port, strict_port)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    print(f"Story Studio starting at http://{host}:{port}")
    app.launch(
        server_name=host,
        server_port=port,
        css=TIP_CSS,
        js=TIP_JS,
    )

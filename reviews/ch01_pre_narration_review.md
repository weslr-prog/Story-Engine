# Pre-Narration Review: Chapter 01

## Goal
- Tune narration voice in the repo-local Gradio UI before batch narration runs.

## Chapter
- Title: Chapter 1: Turning Point
- Voice sample: voices/p233_023.wav
- TTS text file: /Users/wes/Desktop/Story-Engine/chapters/ch01_tts.txt
- Target narration file: /Users/wes/Desktop/Story-Engine/audio/ch01_narration.wav

## Launch The Local Gradio UI
```bash
bash scripts/start_chatterbox_tts_ui.sh
```

## Suggested Starting Controls
- Exaggeration: 0.4
- CFG/Pace: 0.65
- Temperature: 0.72
- Min P: 0.05
- Top P: 1.0
- Repetition penalty: 1.2

## Copy/Paste Preview Text
Here's the revised script for the first act of chapter 1:
Chapter 1: Act 1
Scene 1: Tighten Prose, Remove Repetition, Preserve Plot Facts
(Proses and Prose-Level Logic)
The central mystery is a data breach that threatens the security of a highly secretive and profitable financial institution. The scene-level detail is the intricate process of unravelling the evidence that reveals the breach, and the costly consequences of the data breach. The key events are the conflicting evidence, the risky decision, and the revealing data visualizations.

## Approval Step
When the voice sounds right, approve this chapter so the pipeline can continue:
```bash
touch reviews/ch01_pre_narration.approved
```

Then rerun the pipeline or chapter narration command.

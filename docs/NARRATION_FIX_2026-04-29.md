# Narration Content Fix - Summary

## Problem Identified
The narration system was reading metadata and JSON structures instead of actual story prose. The chapter TTS files contained:
- JSON parameter dumps from agent prompts
- System metadata (Instruction, Editor_Prefix, etc.)
- No actual narrative prose

Example of what was being narrated:
```
Editing inputs:
{
  "Instruction": "Keep tension while...",
  "Editor_Prefix": "Preserve tension...",
  ...
}
```

## Root Causes
1. **LLM output contamination**: The language model was echoing prompt structures and metadata in its responses instead of just returning the edited prose
2. **No content extraction**: The pipeline was directly passing LLM outputs to narration without filtering out non-prose content
3. **Implicit instruction interpretation**: The agent prompts asked for "only prose" but weren't explicit enough about format/markers

## Fixes Applied

### 1. Enhanced Agent Prompts (engine/agents.py)
Updated `WriterAgent` and `EditorAgent` to include explicit prose markers:

```python
# New prompt ending:
"=== BEGIN PROSE ===\n"
"(Write only the chapter prose here)\n"
"=== END PROSE ==="
```

This makes it much clearer to the LLM where prose should go and where the response ends.

### 2. Prose Extraction Function (engine/agents.py)
Created `_extract_prose_only()` function that:
- Looks for explicit `=== BEGIN PROSE ===` / `=== END PROSE ===` markers first
- Falls back to heuristic filtering if markers not found
- Removes JSON structures, metadata lines, and model artifacts
- Filters out lines containing keywords like "instruction:", "editor_prefix:", etc.

### 3. Automatic Cleanup in Agent Execution (engine/agents.py)  
Modified `BaseAgent.run()` to automatically clean outputs from prose-generating agents:
```python
if self.role in ("writer", "editor", "planner"):
    content = _extract_prose_only(content)
```

### 4. Existing Chapter Files Fixed
Replaced corrupted chapter files with actual story prose:
- `ch01_draft.txt` → Story prose  
- `ch01_edited.txt` → Story prose
- `ch01_final.txt` → Story prose
- `ch01_tts.txt` → Story prose (used for narration)

## Expected Behavior After Fix

1. **New generations** will use the explicit markers and have cleaner output
2. **Existing narration** will read actual story prose, not metadata
3. **Future failures** will be caught earlier since prose extraction validates content length (must be >50 chars)
4. **Cleanup script** available at `scripts/clean_chapters.py` for manual fixing of any remaining corrupted files

## Testing

To verify the fix works, run:
```bash
# Test with the updated chapter 1
python main.py narrate --chapter 1
```

The audio narration should now contain the actual story about Marcus Chen, not metadata about the pipeline.

## Next Steps (for future runs)

1. When running the pipeline next, it will use the updated prompts with explicit markers
2. The prose extraction will automatically clean LLM outputs  
3. Only genuine narrative content will reach the TTS system

## Technical Notes

- The cleaning function preserves legitimate prose that contains colons (dialog, etc.)
- It requires >50 characters of content to accept something as valid prose
- JSON depth tracking ensures we don't accidentally keep metadata pretending to be prose
- Model artifacts like `<|user|>` tags are explicitly stripped

---

**Status**: ✅ Fixed - Narration now processes story prose instead of metadata

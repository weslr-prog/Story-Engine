# Prompt Reference for Section 5.1

This document is the dedicated prompt/input guide for section 5.1 in the first-story tutorial.

Use it to build clean first-run inputs for:
- story_bible.json
- characters.json
- chapter_briefs.json

## 1. Why These Files Matter

The pipeline uses these three files as the canonical story contract before any chapter drafting.

- story_bible.json: world constraints and narrative direction
- characters.json: stable character behavior and dialogue style
- chapter_briefs.json: per-chapter execution targets

If these are vague or contradictory, prompt tuning in genre packs cannot fully compensate.

## 2. story_bible.json Reference

## Required intent

- Define one clear premise and conflict
- Lock high-level themes
- Add world facts that should never be violated

## Recommended schema

```json
{
  "title": "Glass Meridian",
  "logline": "A forensic archivist uncovers a city-wide memory rewrite tied to her vanished sister.",
  "themes": [
    "truth",
    "identity",
    "cost of certainty"
  ],
  "world_facts": [
    "The city of Meridian records civic memory snapshots nightly.",
    "Unauthorized memory edits are treated as terrorism."
  ]
}
```

## Field guidance

- title: short and specific
- logline: one sentence, protagonist plus pressure plus stakes
- themes: 2 to 5 abstract concepts
- world_facts: immutable rules the writer should preserve

## 3. characters.json Reference

## Required intent

- Give each recurring character a stable voice and motivation
- Keep role and conflict value obvious

## Recommended schema

```json
[
  {
    "name": "Mara Quill",
    "role": "Protagonist",
    "core_wound": "She failed to protect her sister during a memory raid.",
    "flawed_belief": "Control is safer than trust.",
    "voice_style": "Precise, restrained, observational under stress."
  },
  {
    "name": "Ivo Vale",
    "role": "Antagonist",
    "core_wound": "He survived social collapse and now fears public instability.",
    "flawed_belief": "People need managed truth to stay civilized.",
    "voice_style": "Calm, persuasive, strategic, never openly frantic."
  }
]
```

## Field guidance

- name: stable identifier used across chapters
- role: protagonist, antagonist, ally, rival, etc.
- core_wound: origin pain that drives choices
- flawed_belief: what they wrongly believe
- voice_style: dialogue and narration flavor cues

## 4. chapter_briefs.json Reference (First 3 Chapters)

## Required intent

- Keep each chapter focused on one main turn
- Make chapter goals short enough for an early smoke run

## Recommended schema

```json
[
  {
    "chapter_number": 1,
    "title": "Turning Point",
    "goal": "Mara receives contradictory evidence about her sister and chooses to pursue a dangerous lead.",
    "must_include": [
      "One contradiction in evidence",
      "One risky decision",
      "A consequence-driven cliffhanger"
    ],
    "target_words": "500-900"
  },
  {
    "chapter_number": 2,
    "title": "Pressure Gradient",
    "goal": "Mara tests the lead, discovers surveillance pressure, and sacrifices one safe option.",
    "must_include": [
      "A false lead that costs time",
      "Escalating institutional threat",
      "A harder next-step choice"
    ],
    "target_words": "500-900"
  },
  {
    "chapter_number": 3,
    "title": "Signal Debt",
    "goal": "Mara secures partial proof but triggers retaliation that redefines the conflict.",
    "must_include": [
      "A partial reveal",
      "A personal cost",
      "A forward hook into chapter 4"
    ],
    "target_words": "500-900"
  }
]
```

## Field guidance

- chapter_number: integer in ascending order
- title: concise, distinct, no duplicates
- goal: one sentence action objective
- must_include: concrete beats that must appear
- target_words: compact range for early validation

## 5. Quality Checklist Before Running

- JSON parses cleanly
- No contradictory world facts
- Character voice styles are distinct
- Every chapter brief has a decision and a consequence
- Chapter 1 to 3 goals are short and testable

## 6. How This Connects to Prompt Tuning (5.2)

At runtime, these files define content constraints while the genre pack defines writing behavior.

- Content contract: story_bible.json, characters.json, chapter_briefs.json
- Style/control contract: genre_packs/thriller_scifi.yaml

If output quality slips, fix contract first, then tune genre prompts.

## 7. Fast Validation Commands

```bash
python -c "import json; json.load(open('story_bible.json')); print('story_bible.json OK')"
python -c "import json; json.load(open('characters.json')); print('characters.json OK')"
python -c "import json; json.load(open('chapter_briefs.json')); print('chapter_briefs.json OK')"
```

## 8. Prompt Output Compatibility (Phase 1-4)

If you generate source docs using the Update Story Pipeline prompt workflow, validate those text outputs before import/conversion.

Expected files (in `The Last Signal/` by default):
- `Phase 1 - Story DNA Summary.txt`
- `Phase 2 - Story Bible.txt`
- `Phase 3 - Chapter Blueprint.txt`
- optional: `Phase 4 - Writing Prompts.txt`

Run validator:

```bash
python scripts/validate_phase_docs.py --source-dir "The Last Signal" --require-phase4
```

Or run one-step ingest + conversion into Story-Engine root files:

```bash
python scripts/ingest_prompt_outputs.py --source-dir "The Last Signal" --out-dir . --require-phase4
```

Phase 3 strict labels checked by validator:
- `Word target`
- `POV`
- `SCENE ZERO`
- `SCENE BREAKDOWN`
- `CHARACTER BEAT`
- `ACTION BEAT`
- `EMOTIONAL BEAT`
- `INTERIORITY BEAT`
- `CLIFFHANGER`

If validation fails, normalize labels in the source text and rerun conversion.

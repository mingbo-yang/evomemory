"""Task instructions shared by refinement implementations; no model imports."""

INSTRUCTIONS = {
    "wmt19_en_zh": "Correct mistranslations, omissions, names and numbers. Preserve meaning and avoid unnecessary expansion.",
    "wmt19_zh_en": "Correct mistranslations, omissions, names and numbers. Preserve meaning and avoid unnecessary expansion.",
    "coedit_gec": "Correct grammatical errors with minimal edits. Preserve the author's meaning and avoid paraphrasing correct text.",
    "gigaword": "Improve factual accuracy and coverage of the main point. Keep the summary concise and do not add unsupported details.",
}

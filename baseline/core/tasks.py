from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .metrics import task_metric
from .types import TaskExample


def _strip_json_fence(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    return s


def parse_json_object(text: str) -> Dict[str, Any]:
    s = _strip_json_fence(text)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


class TaskAdapter:
    task: str

    def __init__(self, task: str):
        self.task = task

    def system_prompt(self) -> str:
        if self.task.startswith("wmt19"):
            return "You are a professional translator. Follow the user's instruction exactly."
        if self.task == "coedit_gec":
            return "You are a professional English grammar editor. Preserve meaning and make minimal necessary corrections."
        if self.task == "gigaword":
            return "You are a professional news summarization editor. Produce concise and factual summaries."
        return "You are a helpful assistant."

    def initial_prompt(self, example: TaskExample) -> str:
        if self.task == "wmt19_en_zh":
            return f"Translate the following text into Chinese. Output only the translation.\n\nEnglish: {example.source}\nChinese:"
        if self.task == "wmt19_zh_en":
            return f"Translate the following text into English. Output only the translation.\n\nChinese: {example.source}\nEnglish:"
        if self.task == "coedit_gec":
            return f"Correct the grammatical errors in the following sentence. Preserve the original meaning and output only the corrected sentence.\n\nSentence: {example.source}\nCorrected:"
        if self.task == "gigaword":
            return f"Summarize the following news text in one concise sentence. Output only the summary.\n\nText: {example.source}\nSummary:"
        raise ValueError(self.task)

    def direct_refine_prompt(self, example: TaskExample, candidate: str) -> str:
        constraints = {
            "wmt19_en_zh": "Keep the source meaning, fix omissions, mistranslations, unnatural wording, and do not add unsupported information.",
            "wmt19_zh_en": "Keep the source meaning, fix omissions, mistranslations, unnatural wording, and do not add unsupported information.",
            "coedit_gec": "Make minimal necessary grammatical corrections, preserve meaning, and avoid rewriting correct content.",
            "gigaword": "Improve factual consistency, include salient information, remove redundancy, and do not introduce unsupported facts.",
        }[self.task]
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Previous candidate:\n{candidate}\n\n"
            "Treat the previous candidate as unverified. "
            f"{constraints}\n"
            "Output only the revised final text."
        )

    def feedback_prompt(self, example: TaskExample, candidate: str) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Current output:\n{candidate}\n\n"
            "Give specific, actionable feedback for improving the current output. "
            "Do not rewrite the final answer. Do not mark correct content as wrong. "
            "Distinguish omissions, incorrect content, redundancy, and unnatural expression. "
            "Return JSON with keys needs_revision and issues."
        )

    def feedback_refine_prompt(self, example: TaskExample, candidate: str, feedback: str, history: Any = None) -> str:
        history_text = ""
        if history:
            history_text = f"\nPrevious candidates and feedback:\n{history}\n"
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Current output:\n{candidate}\n\n"
            f"Feedback:\n{feedback}\n"
            f"{history_text}\n"
            "Revise the current output using only well-supported feedback. Preserve correct content. Output only the revised final text."
        )

    def checklist_prompt(self, example: TaskExample) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            "Create 2 to 8 concise Yes/No checklist questions for evaluating whether an output satisfies this task. "
            "Return JSON: {\"items\": [{\"id\": \"c1\", \"question\": \"...\"}]}."
        )

    def checklist_eval_prompt(self, example: TaskExample, candidate: str, checklist: List[Dict[str, str]]) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Output to evaluate:\n{candidate}\n\n"
            f"Checklist:\n{json.dumps(checklist, ensure_ascii=False)}\n\n"
            "Evaluate every checklist item strictly. Return JSON: "
            "{\"verdicts\": [{\"item_id\": \"c1\", \"verdict\": \"YES\" or \"NO\", \"reason\": \"...\"}]}."
        )

    def checklist_refine_prompt(self, example: TaskExample, candidate: str, failed_items: List[Dict[str, str]]) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Current output:\n{candidate}\n\n"
            f"Failed checklist items:\n{json.dumps(failed_items, ensure_ascii=False)}\n\n"
            "Revise only to satisfy the failed checklist items. Output only the revised final text."
        )

    def pdr_workspace_prompt(self, example: TaskExample, drafts: List[str]) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Draft outputs:\n{json.dumps(drafts, ensure_ascii=False)}\n\n"
            "Distill these drafts into a bounded workspace. Do not write the final answer. Return JSON with keys "
            "agreements, candidate_disagreements, likely_errors, missing_information, preserve, next_revision_actions."
        )

    def pdr_final_prompt(self, example: TaskExample, workspace: str) -> str:
        return (
            f"Original task:\n{self.initial_prompt(example)}\n\n"
            f"Workspace:\n{workspace}\n\n"
            "Use the workspace to produce one final answer. Output only the final text."
        )

    def tear_estimate_prompt(self, example: TaskExample, candidate: str) -> str:
        return (
            f"Source text:\n{example.source}\n\n"
            f"Translation:\n{candidate}\n\n"
            "Estimate translation errors using MQM. Return JSON with no_error and errors. "
            "Each error should include severity, category, span, and explanation. "
            "Categories may include accuracy/omission, accuracy/mistranslation, accuracy/addition, fluency/grammar, terminology, style/awkward, no_error."
        )

    def tear_refine_prompt(self, example: TaskExample, candidate: str, mqm_feedback: str) -> str:
        direction = "Chinese" if self.task == "wmt19_en_zh" else "English"
        return (
            f"Source text:\n{example.source}\n\n"
            f"Current translation:\n{candidate}\n\n"
            f"MQM feedback:\n{mqm_feedback}\n\n"
            f"Fix the listed errors. Do not add unsupported content or omit source content. Output only the revised {direction} translation."
        )

    def parse_output(self, raw_text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", str(raw_text), flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
        for marker in ["### SPLIT ###", "Refined Translation:", "Refined Summary:", "Corrected:", "Summary:", "Chinese:", "English:"]:
            if marker in text:
                text = text.split(marker)[-1].strip()
        return text.strip().strip('"')

    def reference_utility(self, example: TaskExample, output: str) -> float:
        return task_metric(self.task, example.reference, output)


def get_adapter(task: str) -> TaskAdapter:
    return TaskAdapter(task)


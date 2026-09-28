#!/usr/bin/env python3
"""Disagreement Resolution Pipeline"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR  # Assigned after loading config/env in main()
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR  # Assigned after loading config/env in main()

# Loaded at runtime inside main()
OPENROUTER_API_KEY: Optional[str] = None

CONSULTANT_SYSTEM_PROMPT = ""
JUDGE_SYSTEM_PROMPT_TEMPLATE = ""

CONSULTANT_RANDOM_SEED = 42
MAX_RESPONSE_RETRIES = 3
LOG_PROMPTS = False
PROMPT_LOGS: List[Dict[str, Any]] = []
PROMPT_LOG_PATH: Optional[Path] = None


@dataclass
class ConsultantState:
    label: str
    model_name: str
    messages: List[Dict[str, str]]
    latest_answer: str
    latest_reasoning: List[str]
    initial_answer: str
    initial_reasoning: List[str]
    conflict_history: List[Dict[str, Any]] = field(default_factory=list)
    turns_taken: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "model": display_model_name(self.model_name),
            "initial_answer": self.initial_answer,
            "initial_reasoning": self.initial_reasoning,
            "latest_answer": self.latest_answer,
            "latest_reasoning": self.latest_reasoning,
            "turns_taken": self.turns_taken,
            "conflict_history": self.conflict_history,
        }


def load_env_config(env_path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    if not env_path.exists():
        return config

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def format_choices_text(raw: Any) -> str:
    """Convert the choices blob into a multi-line string."""
    if isinstance(raw, dict):
        return "\n".join(f"{label}. {text}" for label, text in raw.items())

    text = str(raw or "").strip()
    if not text:
        return "Choices unavailable."

    pattern = re.compile(r"([A-D])[\.\)]\s*")
    parts = pattern.split(text)
    if len(parts) > 1:
        combined: List[str] = []
        current_label = None
        for token in parts:
            if not token:
                continue
            if len(token) == 1 and token in "ABCD":
                current_label = token
                continue
            if current_label:
                clean_token = token.strip().replace("\n", " ")
                combined.append(f"{current_label}. {clean_token}")
                current_label = None
        if combined:
            return "\n".join(combined)

    # Fallback: split by comma/newlines.
    tokens = [seg.strip() for seg in re.split(r"[\n,]+", text) if seg.strip()]
    return "\n".join(tokens) if tokens else text


def sanitize_filename_component(text: str) -> str:
    clean = text.strip().replace("/", "_").replace("\\", "_")
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", clean)
    return clean or "unknown"


def display_model_name(model_identifier: str) -> str:
    if not model_identifier:
        return "unknown"
    identifier = model_identifier.strip()
    if "/" in identifier:
        identifier = identifier.rsplit("/", 1)[-1]
    return identifier or "unknown"


def normalize_reasoning(reasoning: Any) -> List[str]:
    if isinstance(reasoning, list):
        return [str(step).strip() for step in reasoning if str(step).strip()]
    if isinstance(reasoning, str):
        return [seg.strip() for seg in reasoning.split("\n") if seg.strip()]
    return []


def render_reasoning(reasoning: List[str]) -> str:
    if not reasoning:
        return "No reasoning provided."
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(reasoning))


def load_model_data(model1_path: Path, model2_path: Path) -> List[Dict[str, Any]]:
    with open(model1_path, "r", encoding="utf-8") as handle:
        model1_data = json.load(handle)
    with open(model2_path, "r", encoding="utf-8") as handle:
        model2_data = json.load(handle)

    by_index_1 = {entry["index"]: entry for entry in model1_data}
    by_index_2 = {entry["index"]: entry for entry in model2_data}

    matched: List[Dict[str, Any]] = []
    for idx in sorted(set(by_index_1.keys()) & set(by_index_2.keys())):
        matched.append({"index": idx, "model1": by_index_1[idx], "model2": by_index_2[idx]})
    return matched


def force_json_parse(blob: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(blob)
    except json.JSONDecodeError:
        pass

    start, end = blob.find("{"), blob.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return True, json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            return False, {"raw": blob, "error": "unable to parse json"}
    return False, {"raw": blob, "error": "json boundary not found"}


def call_openrouter(model_name: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter API key not configured. Set API_KEY in agent/.env or OPENROUTER_API_KEY.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=90,
    )
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"OpenRouter error: {body['error']}")
    choices = body.get("choices")
    if not choices:
        raise RuntimeError("OpenRouter response missing choices payload.")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content or "")

    if not text.strip():
        raise RuntimeError("OpenRouter returned an empty message.")
    return text


def fetch_json_response_with_retries(
    model_name: str,
    messages: List[Dict[str, str]],
    *,
    case_index: int,
    actor_label: str,
    sleep_seconds: float,
    fallback_messages: Optional[List[Dict[str, str]]] = None,
    log_fn: Optional[Callable[[List[Dict[str, str]]], None]] = None,
) -> Tuple[str, Any]:
    """Call OpenRouter and parse JSON with retries for formatting failures, with optional fallback."""

    def attempt(current_messages: List[Dict[str, str]]) -> Tuple[str, Any]:
        last_error: Optional[Any] = None
        for attempt_index in range(1, MAX_RESPONSE_RETRIES + 1):
            if log_fn:
                log_fn(current_messages)
            raw_response = call_openrouter(model_name, current_messages)
            maybe_sleep(sleep_seconds)
            ok, parsed = force_json_parse(raw_response)
            if ok:
                return raw_response, parsed
            last_error = parsed
            print(
                f"[case {case_index}] WARN: {actor_label} returned invalid format "
                f"(attempt {attempt_index}/{MAX_RESPONSE_RETRIES}): {parsed}"
            )
        print(
            f"[case {case_index}] ERROR: {actor_label} failed to produce valid JSON "
            f"after {MAX_RESPONSE_RETRIES} attempts. Last error: {last_error}"
        )
        raise RuntimeError(f"{actor_label} response was not JSON after {MAX_RESPONSE_RETRIES} attempts.")

    try:
        return attempt(messages)
    except requests.HTTPError as exc:
        if (
            fallback_messages
            and exc.response is not None
            and exc.response.status_code == 400
        ):
            print(
                f"[case {case_index}] WARN: {actor_label} hit HTTP 400; retrying with system prompt merged into user content."
            )
            return attempt(fallback_messages)
        raise


def build_consultant_state(
    label: str,
    model_name: str,
    entry: Dict[str, Any],
    reasoning_prompt_template: str,
) -> ConsultantState:
    question_text = entry.get("Question", "").strip()
    choices_text = format_choices_text(entry.get("choices"))
    reasoning_prompt = reasoning_prompt_template.format(
        question_text=question_text,
        choices_text=choices_text,
    )

    initial_payload = json.dumps(
        {
            "answer": entry.get("model_answer", "").strip(),
            "reasoning": normalize_reasoning(entry.get("model_reasoning")),
        },
        ensure_ascii=False,
    )

    messages = [
        {"role": "user", "content": reasoning_prompt},
        {"role": "assistant", "content": initial_payload},
    ]

    return ConsultantState(
        label=label,
        model_name=model_name,
        messages=messages,
        latest_answer=entry.get("model_answer", "").strip(),
        latest_reasoning=normalize_reasoning(entry.get("model_reasoning")),
        initial_answer=entry.get("model_answer", "").strip(),
        initial_reasoning=normalize_reasoning(entry.get("model_reasoning")),
    )


def summarize_conflicts(state: ConsultantState, limit: int = 2) -> str:
    if not state.conflict_history:
        return "No conflict checklists from this consultant."
    lines: List[str] = []
    for block in state.conflict_history[-limit:]:
        conflicts = block.get("conflicts") or []
        if conflicts:
            conflict_bits = "; ".join(
                f"{item.get('issue', 'unknown')} -> {item.get('winner', 'tie')}"
                for item in conflicts[:3]
            )
        else:
            conflict_bits = "No explicit conflicts recorded."
        confidence = block.get("confidence") or "unknown"
        lines.append(f"Turn {block.get('turn')}: {conflict_bits} (confidence: {confidence})")
    return "\n".join(lines)


def render_full_verification_history(state: ConsultantState) -> str:
    if not state.conflict_history:
        return "No verification history recorded."
    chunks: List[str] = []
    for block in state.conflict_history:
        turn = block.get("turn")
        answer = block.get("answer") or "Unknown"
        confidence = block.get("confidence") or "unknown"
        header = f"Turn {turn}: answer {answer} (confidence: {confidence})"
        raw_payload = (block.get("raw") or "").strip()
        if raw_payload:
            chunks.append(f"{header}\n{raw_payload}")
        else:
            conflicts = block.get("conflicts") or []
            if conflicts:
                conflict_bits = "; ".join(
                    f"{item.get('issue', 'unknown')} -> {item.get('winner', 'tie')}" for item in conflicts
                )
            else:
                conflict_bits = "No conflicts recorded."
            chunks.append(f"{header}\n{conflict_bits}")
    return "\n\n".join(chunks)


def build_combined_verification_history(consultant1: ConsultantState, consultant2: ConsultantState) -> str:
    sections = []
    for state in (consultant1, consultant2):
        sections.append(
            f"{state.label} ({display_model_name(state.model_name)}):\n"
            f"{render_full_verification_history(state)}"
        )
    return "\n\n".join(sections).strip()


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def build_consultant_prompt(
    template: str,
    consultant_label: str,
    other_answer: str,
    other_reasoning: List[str],
) -> str:
    other_reasoning_text = render_reasoning(other_reasoning)
    return template.format(
        consultant_label=consultant_label,
        other_answer=other_answer or "Unknown",
        other_reasoning=other_reasoning_text,
    )


def run_consultant_update(
    case_index: int,
    state: ConsultantState,
    other_state: ConsultantState,
    consultant_template: str,
    sleep_seconds: float,
    other_answer_override: Optional[str] = None,
    other_reasoning_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    history_messages: List[Dict[str, str]] = []
    other_answer = other_answer_override if other_answer_override is not None else other_state.latest_answer
    other_reasoning = (
        other_reasoning_override if other_reasoning_override is not None else other_state.latest_reasoning
    )
    user_prompt = build_consultant_prompt(
        consultant_template,
        consultant_label=state.label,
        other_answer=other_answer,
        other_reasoning=other_reasoning,
    )
    user_message = {"role": "user", "content": user_prompt}
    conversation_base: List[Dict[str, str]] = list(state.messages + history_messages)
    tail_messages: List[Dict[str, str]] = []
    if CONSULTANT_SYSTEM_PROMPT:
        tail_messages.append({"role": "system", "content": CONSULTANT_SYSTEM_PROMPT})
    tail_messages.append(user_message)
    conversation = conversation_base + tail_messages
    fallback_tail = merge_system_into_first_user(tail_messages)
    fallback_conversation = conversation_base + fallback_tail if fallback_tail else None
    raw_response, parsed = fetch_json_response_with_retries(
        state.model_name,
        conversation,
        case_index=case_index,
        actor_label=state.label,
        sleep_seconds=sleep_seconds,
        fallback_messages=fallback_conversation,
        log_fn=lambda msgs: log_prompt(state.label, state.model_name, case_index, msgs),
    )

    state.messages.append(user_message)
    state.messages.append({"role": "assistant", "content": raw_response})

    answer = str(parsed.get("answer", state.latest_answer)).strip().upper()
    reasoning = normalize_reasoning(parsed.get("reasoning"))
    conflicts = parsed.get("conflicts") or []
    confidence = parsed.get("confidence")

    state.turns_taken += 1
    state.latest_answer = answer or state.latest_answer
    state.latest_reasoning = reasoning or state.latest_reasoning
    state.conflict_history.append(
        {
            "turn": state.turns_taken,
            "answer": state.latest_answer,
            "conflicts": conflicts,
            "confidence": confidence,
            "raw": raw_response,
        }
    )

    print(
        f"[case {case_index}] {state.label} -> answer {state.latest_answer} "
        f"(turn {state.turns_taken}, confidence={confidence})"
    )

    return {
        "answer": state.latest_answer,
        "reasoning": state.latest_reasoning,
        "conflicts": conflicts,
        "confidence": confidence,
        "raw": raw_response,
    }


def execute_consultant_turn(
    case_index: int,
    consultant1: ConsultantState,
    consultant2: ConsultantState,
    consultant_template: str,
    sleep_seconds: float,
) -> List[Dict[str, str]]:
    """Run a paired consultant revision turn and return log messages."""
    turn_logs: List[Dict[str, str]] = []
    snapshot_answers = {
        "consultant_1": consultant1.latest_answer,
        "consultant_2": consultant2.latest_answer,
    }
    snapshot_reasoning = {
        "consultant_1": list(consultant1.latest_reasoning),
        "consultant_2": list(consultant2.latest_reasoning),
    }

    for label, state in (("consultant_1", consultant1), ("consultant_2", consultant2)):
        other_label = "consultant_2" if label == "consultant_1" else "consultant_1"
        other_state = consultant2 if label == "consultant_1" else consultant1
        update = run_consultant_update(
            case_index,
            state,
            other_state,
            consultant_template,
            sleep_seconds,
            other_answer_override=snapshot_answers[other_label],
            other_reasoning_override=snapshot_reasoning[other_label],
        )
        turn_logs.append(
            {
                "role": "user",
                "content": (
                    f"{state.label} updated answer to {update['answer']} "
                    f"(confidence={update.get('confidence')})"
                ),
            }
        )

    return turn_logs


def build_judge_replacements(
    question_text: str,
    choices_text: str,
    consultant1: ConsultantState,
    consultant2: ConsultantState,
    turn_usage: Tuple[int, int],
) -> Dict[str, str]:
    turns_used, max_turns = turn_usage
    answers_agree = consultant1.latest_answer == consultant2.latest_answer
    return {
        "{{QUESTION}}": question_text or "Question unavailable.",
        "{{CHOICES}}": choices_text or "Choices unavailable.",
        "{{MODEL_1_ANSWER}}": consultant1.latest_answer or "Unknown",
        "{{MODEL_1_REASONING}}": render_reasoning(consultant1.latest_reasoning),
        "{{MODEL_2_ANSWER}}": consultant2.latest_answer or "Unknown",
        "{{MODEL_2_REASONING}}": render_reasoning(consultant2.latest_reasoning),
        "{{CONSULTANT_TURNS_USED}}": str(turns_used),
        "{{CONSULTANT_TURNS_MAX}}": str(max_turns),
        "{{AGREEMENT_STATUS}}": "agree" if answers_agree else "disagree",
        "{{CONSULTANT_1_CONFLICTS}}": summarize_conflicts(consultant1),
        "{{CONSULTANT_2_CONFLICTS}}": summarize_conflicts(consultant2),
    }


def fill_template(template: str, replacements: Dict[str, str]) -> str:
    formatted = template
    for token, value in replacements.items():
        formatted = formatted.replace(token, value)
    return formatted


def log_prompt(actor_label: str, model_name: str, case_index: int, messages: List[Dict[str, str]]) -> None:
    if not LOG_PROMPTS or PROMPT_LOG_PATH is None:
        return
    PROMPT_LOGS.append(
        {
            "actor": actor_label,
            "model": model_name,
            "case_index": case_index,
            "messages": deepcopy(messages),
        }
    )


def merge_system_into_first_user(messages: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    if not messages or messages[0].get("role") != "system":
        return None
    merged = deepcopy(messages)
    system_entry = merged.pop(0)
    system_text = system_entry.get("content", "").strip()
    if not system_text:
        return merged
    for entry in merged:
        if entry.get("role") == "user":
            entry["content"] = f"{system_text}\n\n{entry.get('content', '')}"
            return merged
    return None


def log_consultant_seed(case_index: int, state: ConsultantState) -> None:
    if not LOG_PROMPTS or PROMPT_LOG_PATH is None:
        return
    seed_messages: List[Dict[str, str]] = []
    if CONSULTANT_SYSTEM_PROMPT:
        seed_messages.append({"role": "system", "content": CONSULTANT_SYSTEM_PROMPT})
    seed_messages.extend(state.messages)
    log_prompt(f"{state.label}_seed", state.model_name, case_index, seed_messages)


def run_judge_review(
    case_index: int,
    judge_model: str,
    base_template: str,
    question_text: str,
    choices_text: str,
    consultant1: ConsultantState,
    consultant2: ConsultantState,
    chat_history: List[Dict[str, str]],
    turn_usage: Tuple[int, int],
    sleep_seconds: float,
) -> Dict[str, Any]:
    replacements = build_judge_replacements(
        question_text,
        choices_text,
        consultant1,
        consultant2,
        turn_usage,
    )
    user_prompt = fill_template(base_template, replacements)
    verification_text = build_combined_verification_history(consultant1, consultant2)
    if verification_text:
        user_prompt = (
            f"{user_prompt}\n\n"
            "Full verification history:\n"
            f"{verification_text}"
        )
    system_prompt = fill_template(JUDGE_SYSTEM_PROMPT_TEMPLATE, replacements)
    user_message = {"role": "user", "content": user_prompt}
    tail_messages = [
        {"role": "system", "content": system_prompt},
        user_message,
    ]
    messages = list(chat_history) + tail_messages
    fallback_tail = merge_system_into_first_user(tail_messages)
    fallback_messages = list(chat_history) + fallback_tail if fallback_tail else None

    raw_response, parsed = fetch_json_response_with_retries(
        judge_model,
        messages,
        case_index=case_index,
        actor_label="judge",
        sleep_seconds=sleep_seconds,
        fallback_messages=fallback_messages,
        log_fn=lambda msgs: log_prompt("judge", judge_model, case_index, msgs),
    )

    chat_history.extend(
        [
            user_message,
            {"role": "assistant", "content": raw_response},
        ]
    )

    selected_answer = (parsed.get("selected_answer") or "").strip()
    if not selected_answer:
        raise RuntimeError(f"Judge did not provide 'selected_answer' for case {case_index}: {parsed}")

    parsed["raw"] = raw_response
    print(
        f"[case {case_index}] Judge finalized with {parsed.get('winner')} "
        f"(selected={selected_answer}, reason={parsed.get('reason')})"
    )
    return parsed


def determine_case_subset(
    matched_cases: List[Dict[str, Any]],
    offset: int,
    limit: Optional[int],
    target_index: Optional[int],
) -> List[Dict[str, Any]]:
    if target_index is not None:
        for case in matched_cases:
            if case["index"] == target_index:
                return [case]
        raise ValueError(f"Case with index {target_index} not found.")

    start = max(offset, 0)
    if limit is None:
        return matched_cases[start:]
    return matched_cases[start : start + limit]


def run_case(
    case: Dict[str, Any],
    models: Dict[str, str],
    prompts: Dict[str, str],
    args: argparse.Namespace,
    swap_consultants: bool,
) -> Dict[str, Any]:
    index = case["index"]
    question_text = case["model1"].get("Question", "").strip()
    choices_text = format_choices_text(case["model1"].get("choices"))
    correct_choice = case["model1"].get("correct_choice") or case["model1"].get("Correct Answer")

    entry_consultant1 = case["model1"]
    entry_consultant2 = case["model2"]
    consultant1_model = models["consultant_1"]
    consultant2_model = models["consultant_2"]
    if swap_consultants:
        entry_consultant1, entry_consultant2 = entry_consultant2, entry_consultant1
        consultant1_model, consultant2_model = consultant2_model, consultant1_model
        print(f"[case {index}] Consultant order randomized (swapped).")

    consultant1 = build_consultant_state(
        "consultant_1",
        consultant1_model,
        entry_consultant1,
        prompts["reasoning"],
    )
    consultant2 = build_consultant_state(
        "consultant_2",
        consultant2_model,
        entry_consultant2,
        prompts["reasoning"],
    )

    chat_history: List[Dict[str, str]] = []
    judge_decisions: List[Dict[str, Any]] = []
    total_consult_turns = 0
    final_decision: Optional[Dict[str, Any]] = None

    def consultants_agree() -> bool:
        return consultant1.latest_answer == consultant2.latest_answer

    # Force consultants to continue iterating until either consensus is reached or
    # the maximum number of turns is exhausted.
    while not consultants_agree() and total_consult_turns < args.max_turns:
        execute_consultant_turn(
            index,
            consultant1,
            consultant2,
            prompts["consultant"],
            args.sleep,
        )
        total_consult_turns += 1

    if not consultants_agree() and total_consult_turns >= args.max_turns:
        print(f"[case {index}] Consultant turn limit reached without consensus; escalating to judge.")
    elif consultants_agree():
        print(f"[case {index}] Consultants reached consensus after {total_consult_turns} turn(s); invoking judge.")

    decision = run_judge_review(
        index,
        models["judge"],
        prompts["judge"],
        question_text,
        choices_text,
        consultant1,
        consultant2,
        chat_history,
        (total_consult_turns, args.max_turns),
        args.sleep,
    )
    judge_decisions.append(decision)
    final_decision = decision

    if final_decision is None:
        raise RuntimeError(f"Judge never finalized case {index}.")

    selected_answer = final_decision.get("selected_answer", "")
    winner_label = (final_decision.get("winner") or "neither").strip()
    return {
        "index": index,
        "question": question_text,
        "choices": choices_text,
        "correct_choice": correct_choice,
        "consultant1_answer": consultant1.latest_answer,
        "consultant2_answer": consultant2.latest_answer,
        "judge_final_answer": selected_answer,
        "judge_winner": winner_label,
        "consultant_1": consultant1.summary(),
        "consultant_2": consultant2.summary(),
        "judge": {
            "model": display_model_name(models["judge"]),
            "decisions": judge_decisions,
            "final_answer": selected_answer,
            "winner": winner_label,
            "reason": final_decision.get("reason"),
        },
        "metadata": {
            "total_consultant_turns": total_consult_turns,
            "max_consultant_turns": args.max_turns,
            "initial_agreement": consultant1.initial_answer == consultant2.initial_answer,
            "consultant_order": "swapped" if swap_consultants else "original",
        },
        "status": "completed",
    }


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_path} ({len(results)} cases).")


def save_prompt_logs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(PROMPT_LOGS, handle, ensure_ascii=False, indent=2)
    print(f"Saved prompt logs to {path} ({len(PROMPT_LOGS)} entries).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the judge-consultant pipeline over GPQA cases.")
    parser.add_argument("--offset", type=int, default=0, help="Start from this matched-case offset.")
    parser.add_argument("--limit", type=int, help="Maximum number of cases to run.")
    parser.add_argument("--case-index", type=int, help="Run a specific dataset index (overrides offset/limit).")
    parser.add_argument("--output", type=Path, default=None, help="Results file path.")
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum consultant revisions allowed.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each API call.")
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="When set, save every prompt payload sent to consultants and the judge.",
    )
    parser.add_argument(
        "--prompt-log-file",
        type=Path,
        help="Optional path for the prompt log JSON (default: <results_dir>/<dataset>/prompt_log.json).",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Resume from an existing results file by retrying any entries with status == 'error'.",
    )
    return parser.parse_args()


def enforce_dataset_filename(raw_value: str, dataset_dir: Path, source_label: str) -> Path:
    filename = raw_value.strip()
    if not filename:
        raise ValueError(f"{source_label} must be a filename (non-empty).")
    path_obj = Path(filename)
    if path_obj.name != filename:
        raise ValueError(f"{source_label} must not include directories: {filename}")
    return dataset_dir / path_obj


def main() -> None:
    args = parse_args()
    env_config = load_env_config(SCRIPT_DIR / ".env")

    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or env_config.get("OPENROUTER_API_KEY")

    dataset_name = os.environ.get("DATASET") or env_config.get("DATASET")
    if not dataset_name:
        raise RuntimeError("DATASET must be provided via environment or .env.")
    dataset_dir = DATA_DIR / dataset_name
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    env_model1 = os.environ.get("CONSULTANT_1_FILENAME") or env_config.get("CONSULTANT_1_FILENAME")
    env_model2 = os.environ.get("CONSULTANT_2_FILENAME") or env_config.get("CONSULTANT_2_FILENAME")
    missing_inputs = []
    if not env_model1:
        missing_inputs.append("CONSULTANT_1_FILENAME")
    if not env_model2:
        missing_inputs.append("CONSULTANT_2_FILENAME")
    if missing_inputs:
        raise RuntimeError("Missing required inputs: " + ", ".join(missing_inputs) + " (set in environment or .env).")

    model1_file = enforce_dataset_filename(env_model1, dataset_dir, "CONSULTANT_1_FILENAME")
    model2_file = enforce_dataset_filename(env_model2, dataset_dir, "CONSULTANT_2_FILENAME")

    prompts_dir_name = (
        os.environ.get("PROMPTS_DIR")
        or os.environ.get("PROMPTS_SET")
        or env_config.get("PROMPTS_DIR")
        or env_config.get("PROMPTS_SET")
    )
    if not prompts_dir_name:
        raise RuntimeError("PROMPTS_DIR or PROMPTS_SET must be set (env or .env) to choose the prompt folder.")

    results_dir_name = os.environ.get("RESULTS_DIR") or env_config.get("RESULTS_DIR")
    if not results_dir_name:
        raise RuntimeError("RESULTS_DIR must be set (env or .env) to choose the results folder.")

    global PROMPTS_DIR, DEFAULT_RESULTS_DIR
    PROMPTS_DIR = SCRIPT_DIR / prompts_dir_name
    DEFAULT_RESULTS_DIR = SCRIPT_DIR / results_dir_name / dataset_name
    env_output = os.environ.get("OUTPUT_FILE") or env_config.get("OUTPUT_FILE")

    models = {
        "consultant_1": os.environ.get("CONSULTANT_1") or env_config.get("CONSULTANT_1"),
        "consultant_2": os.environ.get("CONSULTANT_2") or env_config.get("CONSULTANT_2"),
        "judge": os.environ.get("JUDGE") or env_config.get("JUDGE"),
    }
    missing_models = [key for key, value in models.items() if not value]
    if missing_models:
        raise RuntimeError(f"Missing model identifiers for: {', '.join(missing_models)}")

    if args.output is None and env_output:
        candidate = Path(env_output)
        args.output = candidate if candidate.is_absolute() else DEFAULT_RESULTS_DIR / candidate
    if args.output is None:
        judge_slug = sanitize_filename_component(display_model_name(models["judge"]))
        consultant1_slug = sanitize_filename_component(display_model_name(models["consultant_1"]))
        consultant2_slug = sanitize_filename_component(display_model_name(models["consultant_2"]))
        default_filename = f"{judge_slug}_{consultant1_slug}_{consultant2_slug}.json"
        args.output = DEFAULT_RESULTS_DIR / default_filename

    global LOG_PROMPTS, PROMPT_LOG_PATH
    LOG_PROMPTS = args.log_prompts
    if LOG_PROMPTS:
        if args.prompt_log_file:
            PROMPT_LOG_PATH = (
                args.prompt_log_file
                if args.prompt_log_file.is_absolute()
                else DEFAULT_RESULTS_DIR / args.prompt_log_file
            )
        else:
            PROMPT_LOG_PATH = DEFAULT_RESULTS_DIR / "prompt_log.json"
    else:
        PROMPT_LOG_PATH = None

    prompts = {
        "reasoning": load_prompt(PROMPTS_DIR / "reasoning_generation.txt"),
        "judge": load_prompt(PROMPTS_DIR / "judge.txt"),
        "consultant": load_prompt(PROMPTS_DIR / "consultant.txt"),
    }
    judge_system_prompt = load_prompt(PROMPTS_DIR / "judge_system.txt")

    global CONSULTANT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT_TEMPLATE
    CONSULTANT_SYSTEM_PROMPT = ""
    JUDGE_SYSTEM_PROMPT_TEMPLATE = judge_system_prompt

    matched_cases = load_model_data(model1_file, model2_file)
    display_models = {key: display_model_name(value) for key, value in models.items()}
    print(
        f"Loaded {len(matched_cases)} matched cases from dataset '{dataset_name}' "
        f"using consultants {display_models['consultant_1']} vs {display_models['consultant_2']} "
        f"and judge {display_models['judge']} (source files: {model1_file.name} / {model2_file.name})."
    )

    cached_results: List[Dict[str, Any]] = []
    cached_indices: Dict[int, Dict[str, Any]] = {}
    if args.cached and args.output and args.output.exists():
        try:
            cached_results = json.loads(args.output.read_text(encoding="utf-8"))
            cached_indices = {entry["index"]: entry for entry in cached_results if "index" in entry}
            error_indices = {entry["index"] for entry in cached_results if entry.get("status") == "error"}
            print(
                f"Loaded {len(cached_results)} cached result(s) from {args.output}. "
                f"{len(error_indices)} marked as error."
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: Unable to read cached results from {args.output}: {exc}")
            cached_results = []
            cached_indices = {}
            error_indices = set()
    else:
        error_indices = set()

    cases_to_run = determine_case_subset(matched_cases, args.offset, args.limit, args.case_index)
    if error_indices:
        cases_to_run = [case for case in cases_to_run if case["index"] in error_indices]
        print(f"Retrying {len(cases_to_run)} cached error case(s).")
    else:
        print(f"Running {len(cases_to_run)} case(s) starting at offset {args.offset}.")
    rng = random.Random(CONSULTANT_RANDOM_SEED)
    consultant_role_swaps = {case["index"]: rng.choice((False, True)) for case in cases_to_run}

    results: List[Dict[str, Any]] = []
    for case in cases_to_run:
        try:
            swap_flag = consultant_role_swaps.get(case["index"], False)
            result = run_case(case, models, prompts, args, swap_flag)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[case {case['index']}] ERROR: {exc}")
            results.append(
                {
                    "index": case["index"],
                    "status": "error",
                    "error": str(exc),
                }
            )
        else:
            results.append(result)

    if args.cached and cached_results:
        merged = []
        rerun_indices = {case["index"] for case in cases_to_run}
        for entry in cached_results:
            if entry.get("index") not in rerun_indices:
                merged.append(entry)
        merged.extend(results)
        final_results = sorted(merged, key=lambda item: item.get("index", -1))
    else:
        final_results = results

    save_results(final_results, args.output)
    if LOG_PROMPTS and PROMPT_LOG_PATH:
        save_prompt_logs(PROMPT_LOG_PATH)


if __name__ == "__main__":
    main()

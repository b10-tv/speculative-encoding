"""Everything about text: source/target token alignment, the serving arm's sample settings,
chat rendering and samples per model, and answer grading."""

from __future__ import annotations

import bisect
from typing import Sequence
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import torch
import re
import string
from collections import Counter


# ---- Token alignment ---------------------------------------------------------------------------
# Character-span alignment between a source and a target tokenization.
#
# Both models keep their own tokenizer. For one document, shared character
# offsets give:
#
# * ``limits[t]``: how many source tokens end at or before target token ``t``
#   ends. Target token ``t`` may read only source tokens ``< limits[t]``;
# * ``positions[s]``: the position of source token ``s`` in the target's token
#   frame, from its character midpoint (used only for rotary positions).
#
# Zero-width spans (special tokens) sit at the end of the text seen so far.

Span = tuple[int, int]


def _spans(spans: Sequence[Span]) -> list[Span]:
    output, seen = [], 0
    for start, end in spans:
        start, end = int(start), int(end)
        if end < start:
            raise ValueError("character spans must have end >= start")
        if end == start:
            start = end = max(seen, start)
        output.append((start, end))
        seen = max(seen, end)
    return output


def _check_non_decreasing(values: Sequence[float], name: str) -> None:
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be non-decreasing")


def causal_limits(
    source_spans: Sequence[Span], target_spans: Sequence[Span]
) -> list[int]:
    """Per target token, the number of source tokens that end at or before it
    ends."""

    ends = [end for _, end in _spans(source_spans)]
    _check_non_decreasing(ends, "source span ends")
    return [bisect.bisect_right(ends, end) for _, end in _spans(target_spans)]


def fractional_positions(
    source_spans: Sequence[Span], target_spans: Sequence[Span]
) -> list[float]:
    """Source-token positions in target-token units, from each source token's
    midpoint."""

    target = _spans(target_spans)
    if not target:
        raise ValueError("fractional positions need at least one target span")
    starts = [start for start, _ in target]
    _check_non_decreasing(starts, "target span starts")
    positions = []
    for start, end in _spans(source_spans):
        midpoint = 0.5 * (start + end)
        index = max(0, bisect.bisect_right(starts, midpoint) - 1)
        left, right = target[index]
        fraction = 0.0
        if right > left:
            fraction = min(1.0, max(0.0, (midpoint - left) / (right - left)))
        positions.append(max(0.0, index + fraction - 0.5))
    return positions


# ---- Serving arm -------------------------------------------------------------------------------
# The training arm whose settings serving reads (through ``prompts.make_sample``): the
# values it uses, frozen.

SERVING_ARM = "CURRENT7_W768_S1_CONT6K"
SERVING_CONFIG = {"max_doc_tokens": 8192, "max_target_tokens": 2048, "dense_tokens": 3072}


def arm_config(name, **overrides):
    assert name == SERVING_ARM, f"unknown arm {name!r}: only {SERVING_ARM!r} is kept"
    return {**SERVING_CONFIG, **overrides, "name": name}


# ---- Chat rendering and samples ----------------------------------------------------------------
# Prompts, documents and samples: chat rendering for each model, source/target token
# alignment, and the sample and document tensors the translator reads.

DOCUMENT_SENTINEL = "⟦KVSTILL-DOCUMENT⟧"
ANSWER_SENTINEL = "⟦KVSTILL-ANSWER⟧"
QUESTION_SENTINEL = "⟦KVSTILL-QUESTION⟧"
DEFAULT_SYSTEM = "You are a helpful assistant."
TEMPLATE_DATE = "26 Jul 2024"
GENERATION_TOKENS = {"mcq": 512, "number": 512, "f1": 64, "yesno": 64}
# metric="free": ungraded free-form answers. Not in GENERATION_TOKENS, which the
# eval harness iterates over.
FREE_GENERATION_TOKENS = 512
MCQ_FORMAT_INSTRUCTION = (
    "Respond with brief reasoning in <rationale> tags, then your answer letter "
    "in <answer> tags.\n"
    "Example: <rationale>The text states X.</rationale><answer>A</answer>"
)


@dataclass
class Rendered:
    """One row in one model's token space."""

    header_ids: list
    doc_ids: list
    doc_spans: list
    after_ids: list
    target_ids: list = field(default_factory=list)

    @property
    def native_ids(self):
        return self.header_ids + self.doc_ids + self.after_ids + self.target_ids

    @property
    def doc_start(self):
        return len(self.header_ids)

    @property
    def continuation_ids(self):
        return self.after_ids + self.target_ids


def _template(tokenizer, messages, *, generation):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False,
            date_string=TEMPLATE_DATE)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=generation)


def _encode(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else []


def _encode_offsets(tokenizer, text):
    if not text:
        return [], []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [tuple(map(int, span)) for span in encoded["offset_mapping"]]
    return list(encoded["input_ids"]), spans


@dataclass(frozen=True)
class ChatLayout:
    """Template text around the translated message for one tokenizer and role."""

    header: str
    after: str
    answer_suffix: str


_LAYOUTS = {}


def chat_layout(tokenizer, role):
    """Split a model's chat template at the translated message.

    The question text of system-role rows is template-independent, so it is rendered
    with a placeholder here and substituted afterwards."""

    key = (id(tokenizer), role)
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    if role == "system":
        messages = [{"role": "system", "content": DOCUMENT_SENTINEL},
                    {"role": "user", "content": QUESTION_SENTINEL}]
    elif role == "user":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": DOCUMENT_SENTINEL}]
    else:
        raise ValueError("translated role must be system or user")
    prompt = _template(tokenizer, messages, generation=True)
    answer = [{"role": "assistant", "content": ANSWER_SENTINEL}]
    full = _template(tokenizer, messages + answer, generation=False)
    if prompt.count(DOCUMENT_SENTINEL) != 1 or full.count(ANSWER_SENTINEL) != 1:
        raise ValueError("chat template must render the translated message and answer exactly once")
    before_answer, answer_suffix = full.split(ANSWER_SENTINEL)
    # Some non-thinking generation prompts add an empty thinking block that a saved
    # assistant turn omits (Gemma 4). Keep the exact generation prompt, then use only
    # the template's answer suffix after the answer text.
    if not prompt.startswith(before_answer):
        raise ValueError("chat template generation prompt does not extend the assistant "
                         f"prefix: prompt {prompt!r}, assistant prefix {before_answer!r}")
    header, after = prompt.split(DOCUMENT_SENTINEL)
    _LAYOUTS[key] = ChatLayout(header=header, after=after, answer_suffix=answer_suffix)
    return _LAYOUTS[key]


def raw_prefix_ids(tokenizer):
    """Special tokens a model expects before plain text (for example BOS).

    Some tokenizers (Gemma 4) do not add BOS themselves and rely on the chat template
    to write it. Plain text then gets the BOS that the template opens with."""

    with_special = list(tokenizer("a", add_special_tokens=True)["input_ids"])
    without = list(tokenizer("a", add_special_tokens=False)["input_ids"])
    if with_special[-len(without):] != without:
        return []
    prefix = with_special[: len(with_special) - len(without)]
    bos, bos_id = getattr(tokenizer, "bos_token", None), getattr(tokenizer, "bos_token_id", None)
    if prefix or not bos or bos_id is None or not getattr(tokenizer, "chat_template", None):
        return prefix
    try:
        text = _template(tokenizer, [{"role": "user", "content": "a"}], generation=True)
    except Exception:  # noqa: BLE001 - a template that fails on plain text says nothing
        return prefix
    return [int(tokenizer.bos_token_id)] if text.startswith(bos) else prefix


def render(tokenizer, row, *, document=None, target_text=None, include_document=True):
    """Render ``row`` with ``tokenizer``; ``document`` overrides the row text."""

    document = row["document"] if document is None else document
    if not include_document:
        document = ""
    target_text = row.get("target") if target_text is None else target_text
    if row["kind"] == "raw":
        doc_ids, doc_spans = _encode_offsets(tokenizer, document)
        target_ids = _encode(tokenizer, target_text or "")
        return Rendered(raw_prefix_ids(tokenizer), doc_ids, doc_spans, [], target_ids)
    if row["translated_role"] == "user" and row.get("question"):
        # The user-role layout puts the document in the user turn and has no question
        # slot, so the question would be dropped silently.
        raise ValueError("role='user' rows cannot carry a separate question; "
                         "put it in the document or use role='system'")
    layout = chat_layout(tokenizer, row["translated_role"])
    after = layout.after.replace(QUESTION_SENTINEL, str(row.get("question") or ""))
    doc_ids, doc_spans = _encode_offsets(tokenizer, document)
    target_ids = _encode(tokenizer, target_text + layout.answer_suffix) if target_text else []
    return Rendered(_encode(tokenizer, layout.header), doc_ids, doc_spans,
                    _encode(tokenizer, after), target_ids)


def truncate_to_tokens(tokenizer, text, maximum):
    """Longest prefix of ``text`` ending on a token boundary with ``<= maximum`` tokens."""

    if not text:
        return text
    ids, spans = _encode_offsets(tokenizer, text)
    if len(ids) <= maximum:
        return text
    return text[: spans[maximum - 1][1]]


def chat_row(source, identifier, document, *, question=None, role="system", target=None,
             answer=None, metric="mcq", letters=None, domain=None, **extra):
    return {"id": f"{source}:{identifier}", "source": source, "kind": "chat",
            "translated_role": role, "document": str(document).strip(), "question": question,
            "target": target, "answer": answer, "metric": metric, "letters": letters,
            "domain": domain or source, **extra}


@dataclass
class Question:
    """A question rendered for one tokenizer: its ``after_ids`` follow a document prefix in
    that tokenizer's chat template. ``to_dict()`` is the form saved with encoded items and
    read by both engines; every function that takes questions accepts either form."""

    row_id: str
    metric: str
    after_ids: list[int]
    max_tokens: int
    text: str
    target: str | None = None  # the model whose tokenizer rendered it
    answer: str | None = None
    letters: str | None = None
    continuation_ids: list[int] | None = None  # metric "nll": the reference's ids
    sampling: dict | None = None  # Sampling.to_dict(); None: greedy

    def to_dict(self) -> dict:
        out = {"row_id": self.row_id, "metric": self.metric, "answer": self.answer,
               "letters": self.letters, "after_ids": list(self.after_ids),
               "max_tokens": self.max_tokens, "text": self.text, "target": self.target}
        if self.continuation_ids is not None:
            out["continuation_ids"] = list(self.continuation_ids)
        if self.sampling is not None:
            out["sampling"] = dict(self.sampling)
        return out

    @classmethod
    def from_dict(cls, question: Mapping[str, Any]) -> "Question":
        names = cls.__dataclass_fields__
        return cls(**{name: question[name] for name in names if name in question})


def question_dict(question):
    """``Question`` -> its dict; a dict or text unchanged."""

    return question.to_dict() if isinstance(question, Question) else question


def render_question(tokenizer, text: str, *, metric: str = "f1", row_id: str | None = None,
                    max_tokens: int | None = None, sampling=None, target: str | None = None,
                    **row) -> Question:
    """``text`` as a question for ``tokenizer``'s chat template, sent as given (no MCQ
    scaffolding). ``metric="free"``: ungraded free-form. ``metric="nll"``: scores the
    reference ``answer`` (its mean NLL) instead of generating. ``sampling``
    (``Sampling`` or a dict) travels with the question; default greedy. ``row_id`` keys
    compressed items (default: a digest of ``text``). ``target``: the model the tokenizer
    belongs to (checked when the question is used)."""

    import hashlib

    row_id = row_id or "q" + hashlib.sha1(text.encode()).hexdigest()[:10]
    reference = str(row.get("target") or row.get("answer") or "") if metric == "nll" else ""
    if metric == "nll" and not reference:
        raise ValueError('metric="nll" scores a reference text: pass answer=...')
    chat = chat_row("api", row_id, "", question=text, metric=metric, **row)
    rendered = render(tokenizer, chat, include_document=False, target_text=reference)
    question = Question(
        row_id=chat["id"], metric=metric, after_ids=list(rendered.after_ids),
        max_tokens=generation_tokens(metric, max_tokens), text=text, target=target,
        answer=row.get("answer"), letters=row.get("letters"))
    if metric == "nll":  # one forward over the reference, no generation
        question.continuation_ids, question.max_tokens = list(rendered.continuation_ids), 0
    if sampling is not None:
        from speculative_encoding.serving.hf_engine import as_sampling

        question.sampling = as_sampling(sampling).to_dict()
    return question


def _question_text(question: Mapping[str, Any]) -> str:
    text = str(question["question"])
    if question.get("category") in {"summary", "conversation"}:
        return text
    inline = all(token in text for token in ("A)", "B)", "C)", "D)"))
    choices = question.get("choices")
    if not inline and isinstance(choices, Mapping) and all(c in choices for c in "ABCD"):
        text = f"{text}\n" + "\n".join(f"{letter}) {choices[letter]}" for letter in "ABCD")
    return f"{text}\n\n{MCQ_FORMAT_INSTRUCTION}"


def _answer_text(question: Mapping[str, Any]) -> str | None:
    if question.get("answer_text"):
        return str(question["answer_text"])
    if question.get("verification_text"):
        return str(question["verification_text"])
    answer = question.get("answer")
    if isinstance(answer, str) and answer in "ABCD":
        return f"<answer>{answer}</answer>"
    return None


@dataclass(frozen=True)
class SpanAlignment:
    """Alignment of source tokens into a target token frame."""

    limits: tuple[int, ...]
    positions: tuple[float, ...]


def align(source_spans, target_spans):
    return SpanAlignment(limits=tuple(causal_limits(source_spans, target_spans)),
                         positions=tuple(fractional_positions(source_spans, target_spans)))


@dataclass
class DocumentTensors:
    """Everything the translator needs for one document in one target's frame."""

    source_rows: torch.Tensor
    source_positions: torch.Tensor
    limits: torch.Tensor
    gather: torch.Tensor
    target_positions: torch.Tensor
    target_embeds: torch.Tensor | None
    source_key: str | None = None


def document_tensors(sample, source_rows, reader, device):
    doc_start, alignment = sample.target.doc_start, sample.alignment
    # Each target token gathers the latest source token that ends before it does.
    last_source = [limit - 1 for limit in alignment.limits]
    with torch.no_grad():
        ids = torch.tensor(sample.target.doc_ids, device=device)
        if hasattr(reader, "embed"):
            embeds = reader.embed(ids)
        else:
            embeds = reader.get_input_embeddings()(ids)
    positions = torch.tensor(alignment.positions, device=device, dtype=torch.float32)
    return DocumentTensors(
        source_rows=source_rows,
        source_positions=positions + doc_start,
        limits=torch.tensor(alignment.limits, device=device, dtype=torch.long),
        gather=torch.tensor(last_source, device=device, dtype=torch.long),
        target_positions=torch.arange(doc_start, doc_start + len(sample.target.doc_ids),
                                      device=device),
        target_embeds=embeds,
        source_key=sample.source_key,
    )


@dataclass
class Sample:
    row: dict
    source: Rendered
    target: Rendered
    alignment: SpanAlignment
    kind: str
    window: int = 0
    source_key: str | None = None
    source_rows: torch.Tensor | None = None


def make_sample(row, source_tokenizer, target_tokenizer, *, kind, config, target_text=None,
                document=None):
    original = row["document"] if document is None else document
    document = truncate_to_tokens(source_tokenizer, original, config["max_doc_tokens"])
    if len(document) < len(original):
        warnings.warn(
            f"document truncated to max_doc_tokens={config['max_doc_tokens']} source tokens "
            f"({len(document)} of {len(original)} characters kept); pass a larger "
            "max_doc_tokens to keep it whole", stacklevel=2)
    source = render(source_tokenizer, row, document=document, target_text="")
    target = render(target_tokenizer, row, document=document,
                    target_text=target_text if kind == "task" else "")
    target.target_ids = target.target_ids[: config["max_target_tokens"]]
    if not source.doc_ids or not target.doc_ids:
        return None
    if kind == "task" and not target.continuation_ids:
        return None
    return Sample(row, source, target, align(source.doc_spans, target.doc_spans), kind)


def generation_tokens(metric, max_tokens=None):
    """``max_tokens`` if given, else the metric's default."""

    if max_tokens:
        return int(max_tokens)
    return FREE_GENERATION_TOKENS if metric == "free" else int(GENERATION_TOKENS.get(metric, 512))


def question_text(question: Mapping[str, Any]) -> str:
    """The text a question row carries: plain for ``metric == "free"``, else with the
    paper's MCQ scaffolding."""

    if question.get("metric") == "free":
        return str(question["question"])
    return _question_text(question)


def followup_ids(tokenizer, text, *, after=None, role="system"):
    """Token ids that follow an assistant answer to start the next user turn ``text``
    and the next answer: the template's end of the assistant turn, the user turn and the
    generation prompt (``continue_turn``'s next ``turn_ids``). ``after``: the previous
    answer's ids; when they end with the token that opens this text (the model's
    end-of-turn stop token), that token is not repeated."""

    sentinel = "⟦KVSTILL-NEXT⟧"
    if role == "system":
        first = [{"role": "system", "content": DOCUMENT_SENTINEL}, {"role": "user", "content": "q"}]
    else:
        first = [{"role": "system", "content": DEFAULT_SYSTEM},
                 {"role": "user", "content": DOCUMENT_SENTINEL}]
    turns = [{"role": "assistant", "content": ANSWER_SENTINEL},
             {"role": "user", "content": sentinel}]
    rendered = _template(tokenizer, first + turns, generation=True)
    if rendered.count(ANSWER_SENTINEL) != 1 or rendered.count(sentinel) != 1:
        raise ValueError("chat template must render the answer and the next user turn exactly once")
    ids = _encode(tokenizer, rendered.split(ANSWER_SENTINEL, 1)[1].replace(sentinel, str(text)))
    if after and ids and int(list(after)[-1]) == ids[0]:
        ids = ids[1:]
    return ids


# ---- Grading -----------------------------------------------------------------------------------
# Answer extraction and scoring: multiple choice, number, token F1, yes/no.

def extract_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    if match:
        return match.group(1).strip(), True
    match = re.search(r"<answer>\s*([^\n<]*)", text, re.I)
    if match and match.group(1).strip():
        return match.group(1).strip(), True
    return None, False


def parse_choice(text, letters):
    pattern = "".join(letters)
    for regex in (rf"<answer>\s*([{pattern}])\s*</answer>", rf"<answer>\s*([{pattern}])\b",
                  rf"Answer\s*:?\s*\**\s*([{pattern}])\b"):
        match = re.search(regex, text, re.I)
        if match:
            return match.group(1).upper()
    stripped = text.strip()
    if stripped and stripped[0].upper() in letters and (
            len(stripped) == 1 or not stripped[1].isalpha()):
        return stripped[0].upper()
    return None


def _number(value):
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("$", "").strip().rstrip(".")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = float(match.group(0))
    return int(number) if number.is_integer() else round(number, 6)


def normalize_text(value):
    value = str(value).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction, gold):
    predicted, expected = normalize_text(prediction).split(), normalize_text(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def grade(metric, text, answer, letters=None):
    """Score in [0, 1], whether an answer was parsed, and the prediction."""

    if metric == "mcq":
        prediction = parse_choice(text, letters or "ABCD")
        return {"score": float(prediction == answer), "parsed": prediction is not None,
                "prediction": prediction}
    extracted, parsed = extract_answer(text)
    if metric == "number":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text) or [None]
        prediction = _number(extracted if parsed else numbers[-1])
        score = float(prediction is not None and prediction == _number(answer))
        return {"score": score, "parsed": prediction is not None, "prediction": prediction}
    if metric == "yesno":
        candidate = normalize_text(extracted if parsed else text.strip().split("\n")[0])
        prediction = next((word for word in ("yes", "no") if candidate.startswith(word)), None)
        score = float(prediction == str(answer).lower())
        return {"score": score, "parsed": prediction is not None, "prediction": prediction}
    if metric == "f1":
        candidate = extracted if parsed else text.strip().split("\n")[0]
        golds = answer if isinstance(answer, (list, tuple)) else [answer]
        score = max(token_f1(candidate, gold) for gold in golds)
        return {"score": score, "parsed": parsed, "prediction": candidate}
    raise ValueError(f"unknown metric {metric!r}")


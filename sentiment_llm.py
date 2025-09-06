from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
from annotated_text import annotation
# ---------- Public result type
Label = Literal["Positive", "Negative", "Neutral"]

@dataclass
class SentimentResult:
    label: Label
    confidence: float
    explanation: str
    evidence_phrases: List[str]

# ---------- Prompt materials (compact by design)
LABEL_POLICY = (
    "You evaluate sentiment for MOVIE REVIEWS.\n"
    "Labels: Positive (clear approval or recommendation), Negative (clear disapproval), Neutral (mixed/ambivalent, mainly factual, off-topic, or unclear).\n"
    "Edge rules: Sarcasm with contrast cues → Negative;\n"
    "Negation: 'not bad' → weak Positive; 'not great' → weak Negative;\n"
    "Ratings/emojis: 4-5★/😍 → Positive; 1-2★/🤮 → Negative; 3★/😐 → Neutral unless text leans;\n"
    "Comparatives: choose dominant cue;\n"
    "Target: if sentiment about actors/director implies movie quality, still classify; unrelated topics → Neutral."
)

CONFIDENCE_RUBRIC = (
    "Confidence rubric (map your certainty to [0,1]):\n"
    "0.85-0.95 strong, unambiguous sentiment or 4-5★ / 1-2★ with multiple cues;\n"
    "0.70-0.84 clear but moderate; some hedging;\n"
    "0.55-0.69 mixed signals or mild wording ('okay', 'decent');\n"
    "0.50-0.54 highly ambiguous/neutral/off-topic."
)

EVIDENCE_RULES = (
    "Extract 0-5 short verbatim evidence phrases (≤10 words each) from the review that justify the label.\n"
    "Do NOT invent text; must be substrings of the review."
)
MODE_DIRECTIVES = {
    "Strict": (
        "MODE: STRICT\n"
        "Guideline: When cues are weak or mixed, prefer 'Neutral'. "
        "Choose Positive/Negative only if evidence is clearly dominant."
    ),
    "Lenient": (
        "MODE: LENIENT\n"
        "Guideline: If there's a mild but clear lean overall, choose that label "
        "and avoid Neutral."
    ),
}
JSON_SCHEMA_TEXT = (
    "Return JSON ONLY with keys: label (Positive|Negative|Neutral), confidence (0..1),\n"
    "explanation (<=2 sentences), evidence_phrases (array of 0-5 strings)."
)

# Few-shot exemplars kept brief to control token costs.
FEW_SHOTS = [
    {
        "review": "Absolutely loved it. The pacing was tight and the performances were outstanding! 5/5",
        "out": {
            "label": "Positive",
            "confidence": 0.9,
            "explanation": "Clear praise and 5/5 rating indicate strong approval.",
            "evidence_phrases": ["loved it", "performances were outstanding", "5/5"]
        },
    },
    {
        "review": "It's visually pretty, but the story is dull and I was bored halfway through.",
        "out": {
            "label": "Negative",
            "confidence": 0.82,
            "explanation": "Dominant negative cues outweigh the minor praise.",
            "evidence_phrases": ["story is dull", "bored halfway"]
        },
    },
    {
        "review": "Not bad at all—some jokes land, some don't.",
        "out": {
            "label": "Positive",
            "confidence": 0.7,
            "explanation": "'Not bad' is weak positive despite mixed signals.",
            "evidence_phrases": ["Not bad", "some jokes land"]
        },
    },
    {
        "review": "I describe the plot and actors without any opinion here.",
        "out": {
            "label": "Neutral",
            "confidence": 0.53,
            "explanation": "Mainly factual with no clear sentiment.",
            "evidence_phrases": ["describe the plot", "without any opinion"]
        },
    },
    {
        "review": "Yeah, sure, a 'masterpiece' 🙄. Two hours I'll never get back.",
        "out": {
            "label": "Negative",
            "confidence": 0.86,
            "explanation": "Sarcastic tone and regret indicate disapproval.",
            "evidence_phrases": ["'masterpiece' 🙄", "never get back"]
        },
    },
    {
        "review": "3 stars. Good soundtrack, but overall just okay.",
        "out": {
            "label": "Neutral",
            "confidence": 0.58,
            "explanation": "3★ and balanced comments suggest neutrality.",
            "evidence_phrases": ["3 stars", "overall just okay"]
        },
    },
]

SYSTEM_INSTRUCTION = (
    LABEL_POLICY
    + "\n\n"
    + EVIDENCE_RULES
    + "\n\n"
    + CONFIDENCE_RUBRIC
    + "\n\n"
    + JSON_SCHEMA_TEXT
    + "\nRespond with JSON ONLY."
)

# ---------- Utility: find star/rating hints to pass context
RATING_PATTERNS = [
    re.compile(r"(?P<score>[1-5](?:\.[0-9])?)\s*/\s*5"),  # e.g., 4/5, 3.5/5
    re.compile(r"(?P<stars>[1-5])\s*(?:stars|\u2605|\u2B50|\u272D)", re.IGNORECASE),  # 4 stars, ★, ⭐
]
AUDIT_WEIGHT = 0.20
AUDIT_BAND = {
    "Lenient": (0.55, 0.80),
    "Strict":  (0.50, 0.85),
}

AUDIT_SYSTEM_INSTRUCTION = (
    "You are auditing a sentiment classification of a MOVIE REVIEW.\n"
    "Check if the provided evidence phrases truly support the predicted label, "
    "and whether there are strong counter-cues in the review.\n"
    "Return JSON ONLY with keys:\n"
    "  audit_score (0..1),  # 1 = evidence strongly supports label; 0 = weak/contradictory\n"
    "  reason (<=2 sentences)\n"
    "Keep it concise. JSON only."
)



def segments_with_highlights(text: str, phrases):
    phrases = [p for p in (phrases or []) if isinstance(p, str) and p.strip()]
    if not phrases:
        return [text]
    uniq = sorted(set(phrases), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(p) for p in uniq), re.IGNORECASE)
    segs, i = [], 0
    for m in pattern.finditer(text):
        if m.start() > i:
            segs.append(text[i:m.start()])
        segs.append(annotation(text[m.start():m.end()],"",font_family="Comic Sans MS",border="2px dashed red",))
        i = m.end()
    if i < len(text):
        segs.append(text[i:])
    return segs

def extract_rating_hint(text: str) -> Optional[str]:
    t = text or ""
    for pat in RATING_PATTERNS:
        m = pat.search(t)
        if m:
            if "score" in m.groupdict() and m.group("score"):
                return f"{m.group('score')}/5"
            if "stars" in m.groupdict() and m.group("stars"):
                return f"{m.group('stars')} stars"
    return None


# ---------- Utility: robustly extract first JSON object from a string

def extract_first_json_blob(s: str) -> str:
    dec = json.JSONDecoder()
    for i in range(len(s)):
        try:
            obj, end = dec.raw_decode(s[i:])
            return s[i:i+end]
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found")


# ---------- Validation helpers
ALLOWED_LABELS: Tuple[Label, ...] = ("Positive", "Negative", "Neutral")


def _clip_conf(x: float) -> float:
    if x != x:  # NaN check
        return 0.5
    return max(0.0, min(1.0, float(x)))


def validate_and_sanitize( raw_json: dict, source_text: str, max_evidence: int = 5) -> SentimentResult:
    label = raw_json.get("label", "").strip()
    if label not in ALLOWED_LABELS:
        # Heuristic: map common lowercase or variants
        normalized = label.capitalize()
        label = normalized if normalized in ALLOWED_LABELS else "Neutral"

    conf = _clip_conf(float(raw_json.get("confidence", 0.5)))

    expl = str(raw_json.get("explanation", "")).strip()
    # Keep to max ~2 sentences by truncation if needed
    # Split on sentence enders; keep first two fragments
    parts = re.split(r"(?<=[.!?])\s+", expl)
    if len(parts) > 2:
        expl = " ".join(parts[:2])

    ev = raw_json.get("evidence_phrases", []) or []
    cleaned: List[str] = []
    seen = set()
    for span in ev:
        span = (span or "").strip()
        if not span:
            continue
        # limit to 10 words
        span_words = span.split()
        if len(span_words) > 10:
            span = " ".join(span_words[:10])
        # must be literal substring of source
        if span not in source_text:
            continue
        # dedupe
        if span.lower() in seen:
            continue
        seen.add(span.lower())
        cleaned.append(span)
        if len(cleaned) >= max_evidence:
            break

    return SentimentResult(
        label=label, confidence=conf, explanation=expl, evidence_phrases=cleaned
    )


# ---------- Core classifier
class MovieSentimentClassifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_output_tokens: int = 256,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Export it before running.")

        self.model_name = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            model_name=self.model_name, system_instruction=SYSTEM_INSTRUCTION
        )
    @staticmethod
    def _normalize_review(text: str) -> str:
        if not text:
            return ""
        # Normalize whitespace; keep emojis/punctuation
        t = re.sub(r"\s+", " ", text.strip())
        # Hard cap to keep token use low: keep first and last 800 chars if too long
        max_chars = 1600
        if len(t) > max_chars:
            head, tail = t[:800], t[-800:]
            t = head + " … " + tail
        return t

    def _build_user_prompt(self, review_text: str, mode: Literal["Strict", "Lenient"] = "Strict") -> str:
        rating = extract_rating_hint(review_text)
        hint = f"\nRATING_HINT: {rating}" if rating else ""
        mode_text = MODE_DIRECTIVES.get(mode, MODE_DIRECTIVES["Strict"])
        return (
            "TASK: Classify the following movie review. "
            "Follow the label policy, confidence rubric, and evidence rules.\n\n"
            f"{mode_text}\n\n"
            f"REVIEW:\n{review_text}{hint}\n\n"
            "Respond with JSON only."
        )
    def _build_audit_prompt(
        self,
        review_text: str,
        first_pass: SentimentResult,
        mode: Literal["Strict", "Lenient"]
    ) -> str:
        mode_text = MODE_DIRECTIVES.get(mode, MODE_DIRECTIVES["Strict"])
        fp_json = {
            "label": first_pass.label,
            "confidence": first_pass.confidence,
            "explanation": first_pass.explanation,
            "evidence_phrases": first_pass.evidence_phrases,
        }
        return (
            "TASK: Audit the following first-pass classification against the review.\n"
            "Confirm whether the evidence is sufficient for the label and note counter-cues if any.\n\n"
            f"{mode_text}\n\n"
            "REVIEW:\n" + review_text + "\n\n"
            "FIRST_PASS_JSON:\n" + json.dumps(fp_json, ensure_ascii=False) + "\n\n"
            "Respond with JSON only."
        )

    def _audit_and_nudge_confidence(
            self,
            review_text: str,
            first_pass: SentimentResult,
            mode: Literal["Strict", "Lenient"]
        ) -> float:
            # Build an audit model (reuse same family by default)
            audit_model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=AUDIT_SYSTEM_INSTRUCTION
            )
            prompt = self._build_audit_prompt(review_text, first_pass, mode)
            resp = audit_model.generate_content(
                prompt,
                generation_config=self._genai.types.GenerationConfig(
                    temperature=0.1, top_p=0.95, max_output_tokens=192
                ),
            )
            raw = resp.text or ""
            blob = extract_first_json_blob(raw)
            data = json.loads(blob)

            # audit_score ∈ [0,1]; map to shrink ∈ [0.5, 1.0]
            audit_score = float(data.get("audit_score", 1.0))
            if audit_score != audit_score:  # NaN guard
                audit_score = 1.0
            audit_score = max(0.0, min(1.0, audit_score))
            shrink = 0.5 + 0.5 * audit_score

            raw_conf = float(first_pass.confidence)
            nudged = 0.5 + (raw_conf - 0.5) * shrink
            final_conf = (1.0 - AUDIT_WEIGHT) * raw_conf + AUDIT_WEIGHT * nudged
            return max(0.0, min(1.0, final_conf))
# --- change classify(...) to accept mode and pass it through ---
    def classify(self, review_text: str, mode: Literal["Strict", "Lenient"] = "Strict") -> SentimentResult:
        """Run a single-shot classification and return a validated result."""
        text = self._normalize_review(review_text)

        # Build compact few-shots inline to reinforce JSON-only behavior
        fewshot_block_lines = []
        for ex in FEW_SHOTS:
            fewshot_block_lines.append("Example Review: " + ex["review"])
            fewshot_block_lines.append("Expected JSON: " + json.dumps(ex["out"], ensure_ascii=False))
        fewshot_block = "\n".join(fewshot_block_lines)

        user_prompt = self._build_user_prompt(text, mode=mode)
        prompt = fewshot_block + "\n\n" + user_prompt

        resp = self._model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=self.temperature, top_p=self.top_p, max_output_tokens=self.max_output_tokens
            ),
        )
        raw_text = resp.text or ""
        blob = extract_first_json_blob(raw_text)
        data = json.loads(blob)
        result = validate_and_sanitize(data, source_text=text)

        # --- Always-on policy audit with band gating ---
        band_lo, band_hi = AUDIT_BAND.get(mode, AUDIT_BAND["Strict"])
        if band_lo <= result.confidence <= band_hi:
            # run audit and nudge confidence toward 0.5 if evidence is weak
            result.confidence = self._audit_and_nudge_confidence(text, result, mode)

        return result


# ---------- Optional CLI usage for quick tests
if __name__ == "__main__":
    import sys

    review = " ".join(sys.argv[1:]).strip()
    if not review:
        print("Usage: python sentiment_llm.py <review text>")
        raise SystemExit(1)

    clf = MovieSentimentClassifier()
    out = clf.classify(review)
    print(json.dumps(out.__dict__, ensure_ascii=False, indent=2))

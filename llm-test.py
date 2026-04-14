import json
import mimetypes
import os
import sys
from collections import Counter
import random
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types
from google.genai.errors import APIError
from tqdm import tqdm

# Some environments patch `print` to route through `tqdm.write`, which does not accept `flush=`.
def _tqdm_write(s: str, *, file=None, end="\n", nolock=False, **kwargs: Any) -> None:
    kwargs.pop("flush", None)
    tqdm.write(s, file=file, end=end, nolock=nolock)


# Gemini API (free tier): https://ai.google.dev/gemini-api/docs/quickstart
# Rate limits (RPM, TPM, RPD) are per project; see https://ai.google.dev/gemini-api/docs/rate-limits
# and your caps in AI Studio: https://aistudio.google.com/rate-limit
# Default client throttle targets ~4 RPM (sliding 60s window + min gap); override via GEMINI_MAX_RPM /
# GEMINI_MIN_INTERVAL_SEC if your quota allows more.

IMAGE_DIR = Path("./dashboards")
OUTPUT_FILE = Path("ibcs_compliance_report.json")
COMPLIANCE_CHARTS_FILE = Path("ibcs_compliance_charts.png")
SEED = 42
# Vision + JSON output can use many tokens (counts toward TPM). Lower if you hit token-per-minute limits.
MAX_OUTPUT_TOKENS = int(
    os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", os.environ.get("OPENROUTER_MAX_TOKENS", "8192"))
)
TEMPERATURE = 0.1
TOP_P = 0.9

# Vision-capable Gemini models. Order = try primary first, then fallbacks on unsupported/404.
# Primary: Gemini 3.1 Flash-Lite (preview) — model code per
# https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
)

# Client-side throttling: ~4 requests/minute (60/4 = 15s minimum spacing + sliding window RPM cap).
# Retries in gemini_audit_response() also go through wait(); lower GEMINI_MIN_INTERVAL_SEC if needed locally.
_DEFAULT_TARGET_RPM = int(
    os.environ.get("GEMINI_MAX_RPM", os.environ.get("OPENROUTER_MAX_RPM", "4"))
)
DEFAULT_MIN_INTERVAL_SEC = float(
    os.environ.get(
        "GEMINI_MIN_INTERVAL_SEC",
        os.environ.get("OPENROUTER_MIN_INTERVAL_SEC", str(60.0 / max(1, _DEFAULT_TARGET_RPM))),
    )
)
DEFAULT_MAX_REQUESTS_PER_MINUTE = max(1, _DEFAULT_TARGET_RPM)
# Extra pause after each successful API call (spread TPM; 0 = off).
GEMINI_POST_REQUEST_PAUSE_SEC = float(os.environ.get("GEMINI_POST_REQUEST_PAUSE_SEC", "0.5"))

# Multiple independent audits per image; `compliant` in the report is aggregated (majority / mean tie-break).
AUDIT_RUNS_PER_IMAGE = int(os.environ.get("GEMINI_AUDIT_RUNS", "5"))

# Vision + large max_output_tokens can exceed several minutes; tqdm stays at 0% until the first call returns.
# (Wall-clock wait for the HTTP client; passed to google-genai as milliseconds, see _http_timeout_ms.)
HTTP_TIMEOUT_SEC = float(
    os.environ.get("GEMINI_HTTP_TIMEOUT_SEC", os.environ.get("OPENROUTER_HTTP_TIMEOUT_SEC", "600"))
)
# Gemini rejects server deadlines under 10s (X-Server-Timeout / deadline).
GEMINI_MIN_HTTP_TIMEOUT_MS = 10_000
MAX_RETRIES = 8
MAX_TIMEOUT_RETRIES = 3
INITIAL_BACKOFF_SEC = 2.0

# If Pillow is installed: shrink long edge before upload (smaller payloads, faster requests). 0 = disable.
IMAGE_MAX_EDGE = int(
    os.environ.get(
        "GEMINI_IMAGE_MAX_EDGE",
        os.environ.get("OPENROUTER_IMAGE_MAX_EDGE", "2048"),
    )
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SYSTEM_PROMPT = """
You are an expert in IBCS (International Business Communication Standards), specifically focused on scaling rules in dashboards.
You must evaluate whether a dashboard is compliant or non-compliant based ONLY on the following rules:
RULE 1 - Axis baseline:
All quantitative charts (especially bar charts) must start at zero unless a justified exception is clearly communicated.
RULE 2 - Consistent scaling:
Charts that are visually or logically comparable must use identical scales (same axis range and intervals).
RULE 3 - Zoom-in requirement:
If data differences are too small to be clearly visible at the same scale, a zoomed-in view must be provided.
RULE 4 - Labelling:
If the column has or hasn't got numeric values on top or inside it.
Chart Naming Convention:
Number charts sequentially from top-left to bottom-right, row by row.
Example: In a 2x2 dashboard, the top-left chart = "chart_1", top-right = "chart_2", bottom-left = "chart_3", bottom-right = "chart_4".
Apply this same logic to any dashboard layout (1x3, 3x2, etc.).
Chart Relatedness Rules:
Two or more charts are considered "related" if they meet ANY of the following criteria:
- (a) They share the same unit of measurement (e.g., both show revenue in USD, both show percentages)
- (b) They represent the same metric across different segments or time periods (e.g., monthly sales vs. quarterly sales)
- (c) They are placed side-by-side or in close visual proximity for comparison purposes
If related charts are detected, RULE 2 (consistent scaling) must be evaluated across all of them.
Zoom-In Handling:
If a zoomed-in panel is already present in the dashboard for a specific chart (i.e., an inset, callout, or secondary view showing a magnified range), treat that chart's zoom-in requirement as already satisfied - do NOT flag it as a RULE 3 violation.
Only flag RULE 3 if no zoom-in panel exists and the data differences are too small to read clearly at the current scale.
Compliance Decision Rules:
- A dashboard is non-compliant if ANY violation with confidence >= 0.6 is found.
- A single violation is sufficient to mark the dashboard as non-compliant.
- Violations with confidence < 0.6 must still be listed in the output but do NOT affect the compliant field - flag them with a note: "low_confidence": true.
- If no violations with confidence >= 0.6 are found, the dashboard is compliant.
Your task:
Given a dashboard image, you must:
1. Identify all charts in the dashboard and label them using the naming convention above
2. Determine which charts are related/comparable using the relatedness rules above
3. For each chart:
   - Estimate axis minimum and maximum
   - Determine if the axis starts at zero
4. Compare scales across all related charts
5. Determine if any chart requires a zoom-in view but does not have one
Output STRICTLY in JSON:
{
  "compliant": true/false,
  "violations": [
    {
      "rule": "axis_baseline | consistent_scaling | zoom_requirement | labelling",
      "description": "clear explanation",
      "charts_involved": ["chart_1", "chart_2"],
      "confidence": 0.0-1.0,
      "low_confidence": true/false
    }
  ],
  "charts_detected": [
    {
      "id": "chart_1",
      "type": "bar/line/other",
      "position": "top-left / top-right / etc.",
      "starts_at_zero": true/false/unknown,
      "estimated_range": [min, max],
      "related_to": ["chart_2", "chart_3"],
      "notes": "short reasoning"
    }
  ],
  "final_explanation": "clear human-readable explanation of the compliance result and any violations found"
}
Important constraints:
- Be conservative: if unsure, say "unknown"
- Do NOT hallucinate exact values - estimate only if clearly visible
- Focus on relative scale consistency, not exact pixel-perfect numbers
- Do NOT assume charts are related unless they meet at least one of the three relatedness criteria
- Think step-by-step internally, but do NOT output your reasoning
- Only return the JSON
- Use strict JSON: double-quoted keys and strings, no trailing commas, no comments
""".strip()


def _parse_comma_separated_models(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def gemini_model_chain() -> list[str]:
    """Primary model first, then fallbacks. All must support image input for this script."""
    chain: list[str] = []
    env_chain = os.environ.get("GEMINI_VISION_MODELS", "").strip()
    if not env_chain:
        env_chain = os.environ.get("OPENROUTER_VISION_MODELS", "").strip()
    if env_chain:
        chain.extend(_parse_comma_separated_models(env_chain))
    else:
        chain.extend(DEFAULT_GEMINI_MODELS)
    override = os.environ.get("GEMINI_MODEL", "").strip()
    if not override:
        override = os.environ.get("OPENROUTER_MODEL", "").strip()
    if override:
        chain = [override] + [m for m in chain if m != override]
    seen: set[str] = set()
    out: list[str] = []
    for m in chain:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


@dataclass
class AuditConfig:
    model: str = ""
    vision_fallback_models: list[str] | None = None
    image_dir: Path = IMAGE_DIR
    output_file: Path = OUTPUT_FILE
    compliance_charts_file: Path = COMPLIANCE_CHARTS_FILE
    seed: int = SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC
    max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE
    post_request_pause_sec: float = GEMINI_POST_REQUEST_PAUSE_SEC
    http_timeout_sec: float = HTTP_TIMEOUT_SEC
    audit_runs_per_image: int = AUDIT_RUNS_PER_IMAGE
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    def __post_init__(self) -> None:
        self.min_interval_sec = max(0.0, float(self.min_interval_sec))
        self.max_requests_per_minute = max(1, int(self.max_requests_per_minute))
        self.post_request_pause_sec = max(0.0, float(self.post_request_pause_sec))
        self.audit_runs_per_image = max(1, int(self.audit_runs_per_image))
        chain = gemini_model_chain()
        if not chain:
            chain = list(DEFAULT_GEMINI_MODELS)
        if not self.model:
            self.model = chain[0]
        if self.vision_fallback_models is None:
            self.vision_fallback_models = [m for m in chain[1:] if m != self.model]


class RateLimiter:
    """Enforces a minimum gap between calls and a sliding-window RPM cap."""

    def __init__(self, min_interval_sec: float, max_requests_per_minute: int) -> None:
        self._min_interval = max(0.0, min_interval_sec)
        self._max_rpm = max(0, max_requests_per_minute)
        self._lock = threading.Lock()
        self._last_call_monotonic: float = 0.0
        self._call_times: list[float] = []

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._min_interval > 0:
                elapsed = now - self._last_call_monotonic
                if elapsed < self._min_interval:
                    time.sleep(self._min_interval - elapsed)
                    now = time.monotonic()

            if self._max_rpm > 0:
                window_start = now - 60.0
                self._call_times = [t for t in self._call_times if t > window_start]
                if len(self._call_times) >= self._max_rpm:
                    sleep_until = self._call_times[0] + 60.0 - now
                    if sleep_until > 0:
                        time.sleep(sleep_until)
                        now = time.monotonic()
                        window_start = now - 60.0
                        self._call_times = [t for t in self._call_times if t > window_start]

            self._call_times.append(time.monotonic())
            self._last_call_monotonic = self._call_times[-1]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)


def validate_image_dir(image_dir: Path) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir.resolve()}. "
            "Create it and add dashboard images before running the script."
        )
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory but got: {image_dir.resolve()}")


def get_image_paths(image_dir: Path) -> list[Path]:
    validate_image_dir(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {image_dir.resolve()}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return image_paths


def _guess_mime(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime:
        return mime
    ext = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _image_bytes_for_api(image_path: Path) -> tuple[bytes, str]:
    """
    Return (raw_bytes, mime) for the API. Optionally downscale with Pillow to keep
    multimodal requests smaller and faster.
    """
    mime = _guess_mime(image_path)
    if IMAGE_MAX_EDGE <= 0:
        return image_path.read_bytes(), mime

    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return image_path.read_bytes(), mime

    with Image.open(image_path) as im:
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        elif im.mode != "RGB" and im.mode != "L":
            im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > IMAGE_MAX_EDGE:
            scale = IMAGE_MAX_EDGE / longest
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            im = im.resize((new_w, new_h), resample)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"


def extract_json_block(response_text: str) -> str:
    cleaned = response_text.strip()
    if "```json" in cleaned:
        return cleaned.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
    if "```" in cleaned:
        return cleaned.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
    return cleaned


def extract_balanced_json_object(s: str) -> str | None:
    """First top-level `{ ... }` with string-aware brace matching."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_audit_json(text: str) -> dict[str, Any]:
    """Try fenced block, full text, then balanced-brace extraction."""
    candidates: list[str] = []
    for part in (extract_json_block(text), text.strip()):
        if part and part not in candidates:
            candidates.append(part)
    bal = extract_balanced_json_object(text)
    if bal and bal not in candidates:
        candidates.append(bal)
    last_err: json.JSONDecodeError | None = None
    for cand in candidates:
        if not cand:
            continue
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _with_heartbeat(label: str, fn: Callable[[], Any]) -> Any:
    """Run fn() in a thread; print a line every ~30s until it returns (for long HTTP/JSON work)."""
    done = threading.Event()
    err: list[BaseException] = []
    out: list[Any] = []

    def worker() -> None:
        try:
            out.append(fn())
        except BaseException as e:
            err.append(e)
        finally:
            done.set()

    def heartbeat() -> None:
        t0 = time.monotonic()
        while not done.wait(30.0):
            _tqdm_write(f"  … {label} still running ({time.monotonic() - t0:.0f}s)")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    t.join()
    if err:
        raise err[0]
    return out[0]


def _is_read_timeout(err: BaseException) -> bool:
    if isinstance(err, TimeoutError):
        return True
    name = type(err).__name__.lower()
    if "timeout" in name:
        return True
    msg = str(err).lower()
    return "timed out" in msg or "timeout" in msg


def _nested_timeout_exc(exc: BaseException) -> BaseException | None:
    """Return the first nested exception that looks like a read/connect timeout."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _is_read_timeout(cur):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def _api_error_suggests_model_fallback(exc: APIError) -> bool:
    """Use next model in chain for unsupported / unknown model IDs."""
    if exc.code == 404:
        return True
    status = (exc.status or "").upper()
    if status == "NOT_FOUND":
        return True
    msg = (exc.message or "").lower()
    if "model" not in msg:
        return False
    return (
        "not found" in msg
        or "not supported" in msg
        or "invalid" in msg
        or "does not exist" in msg
        or "unknown" in msg
    )


def _http_timeout_ms(config: AuditConfig) -> int:
    """google-genai HttpOptions.timeout is in milliseconds (not seconds)."""
    ms = int(max(0.0, float(config.http_timeout_sec)) * 1000)
    return max(ms, GEMINI_MIN_HTTP_TIMEOUT_MS)


def _generate_content_config(config: AuditConfig) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
        seed=config.seed,
        http_options=types.HttpOptions(timeout=_http_timeout_ms(config)),
    )


def gemini_audit_response(
    client: genai.Client,
    image_path: Path,
    config: AuditConfig,
    rate_limiter: RateLimiter,
) -> tuple[str, str]:
    """
    Call Gemini with the dashboard image; return (response_text, model_id_used).
    Retries on 429/timeouts; tries fallback model IDs when the API rejects the model name.
    """
    if not config.api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in the environment before running; "
            "see https://ai.google.dev/gemini-api/docs/quickstart"
        )

    raw_image, mime = _with_heartbeat(
        "Image read (and resize if Pillow)",
        lambda: _image_bytes_for_api(image_path),
    )
    image_part = types.Part.from_bytes(data=raw_image, mime_type=mime)
    gen_config = _generate_content_config(config)

    models_chain = [config.model] + [m for m in (config.vision_fallback_models or []) if m != config.model]
    backoff = INITIAL_BACKOFF_SEC
    last_error: str | None = None
    timeout_streak = 0
    timeout = float(config.http_timeout_sec)
    model_idx = 0

    for attempt in range(MAX_RETRIES):
        rate_limiter.wait()
        if attempt > 0:
            tail = (last_error or "")[:200]
            _tqdm_write(f"  … retry {attempt + 1}/{MAX_RETRIES} (last: {tail})")

        model_id = models_chain[min(model_idx, len(models_chain) - 1)]

        def _call() -> types.GenerateContentResponse:
            return client.models.generate_content(
                model=model_id,
                contents=[image_part],
                config=gen_config,
            )

        try:
            response = _with_heartbeat(
                f"Gemini generate_content ({model_id})",
                _call,
            )
        except APIError as exc:
            last_error = str(exc)
            timeout_streak = 0
            if exc.code == 429 or (exc.status or "").upper() == "RESOURCE_EXHAUSTED":
                wait = min(max(backoff, 2.0), 120.0)
                _tqdm_write(f"  Rate limited (429 / RESOURCE_EXHAUSTED); sleeping {wait:.0f}s…")
                time.sleep(wait)
                backoff = min(backoff * 2, 300.0)
                continue
            if _api_error_suggests_model_fallback(exc) and model_idx + 1 < len(models_chain):
                model_idx += 1
                _tqdm_write(
                    f"  Model {model_id!r} rejected ({exc.code}); trying fallback "
                    f"{models_chain[model_idx]!r}…"
                )
                backoff = INITIAL_BACKOFF_SEC
                continue
            raise RuntimeError(last_error) from exc
        except Exception as exc:
            nested = _nested_timeout_exc(exc)
            if nested is not None:
                last_error = str(exc)
                timeout_streak += 1
                _tqdm_write(
                    f"  Request timed out after {timeout:.0f}s "
                    f"({timeout_streak}/{MAX_TIMEOUT_RETRIES}). "
                    f"Set GEMINI_HTTP_TIMEOUT_SEC to wait longer."
                )
                if timeout_streak >= MAX_TIMEOUT_RETRIES:
                    raise RuntimeError(
                        f"Gemini timed out {timeout_streak} times in a row "
                        f"(timeout={timeout:.0f}s). Increase GEMINI_HTTP_TIMEOUT_SEC "
                        f"or reduce image size / GEMINI_MAX_OUTPUT_TOKENS."
                    ) from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            raise

        timeout_streak = 0
        backoff = INITIAL_BACKOFF_SEC
        text = (response.text or "").strip()
        if not text:
            last_error = "Empty model response (no text in candidates)"
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        return text, model_id

    raise RuntimeError(
        f"Gemini request failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _aggregate_compliant_from_runs(
    successful_runs: list[dict[str, Any]],
) -> tuple[bool | None, dict[str, Any]]:
    """Majority vote on `compliant`; ties break by fraction of True among clear (non-null) votes."""
    votes_true = votes_false = votes_unclear = 0
    for r in successful_runs:
        c = r.get("compliant")
        if c is True:
            votes_true += 1
        elif c is False:
            votes_false += 1
        else:
            votes_unclear += 1
    n_clear = votes_true + votes_false
    meta: dict[str, Any] = {
        "votes_true": votes_true,
        "votes_false": votes_false,
        "votes_unclear": votes_unclear,
        "fraction_compliant": (votes_true / n_clear) if n_clear else None,
    }
    if n_clear == 0:
        meta["decision"] = "no_clear_votes"
        return None, meta
    if votes_true > votes_false:
        meta["decision"] = "majority_compliant"
        return True, meta
    if votes_false > votes_true:
        meta["decision"] = "majority_non_compliant"
        return False, meta
    # Tie on counts: use average of clear votes (0.5 → compliant True if >= 0.5)
    frac = votes_true / n_clear
    meta["decision"] = "tie_break_mean"
    return (frac >= 0.5), meta


def _representative_audit_run(
    successful_runs: list[dict[str, Any]],
    compliant_decision: bool | None,
) -> dict[str, Any]:
    """Prefer a run whose `compliant` matches the aggregated flag; else first successful."""
    for r in successful_runs:
        if compliant_decision is None or r.get("compliant") == compliant_decision:
            return r
    return successful_runs[0]


def generate_audit_single(
    image_path: Path,
    config: AuditConfig,
    rate_limiter: RateLimiter,
    client: genai.Client,
    *,
    run_index: int = 0,
    run_label: str = "",
) -> dict[str, Any]:
    """One API call + parse. `run_index` offsets `seed` so repeated runs are not identical."""
    size_mb = image_path.stat().st_size / (1024 * 1024)
    label = f" {run_label}" if run_label else ""
    _tqdm_write(
        f" Requesting audit{label}: {image_path.name} ({size_mb:.2f} MiB on disk) — "
        f"API timeout {config.http_timeout_sec:.0f}s. "
        f"0% until this file finishes; you should see “still running” heartbeats during the request."
    )
    text, model_used = gemini_audit_response(client, image_path, config, rate_limiter)

    try:
        result = parse_audit_json(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model returned non-JSON (or truncated JSON): {exc}\n"
            f"Preview: {text[:800]!r}"
        ) from exc

    result["filename"] = image_path.name
    result["gemini_model"] = model_used
    result["audit_run_index"] = run_index
    if config.post_request_pause_sec > 0:
        time.sleep(config.post_request_pause_sec)
    return result


def generate_audit(
    image_path: Path,
    config: AuditConfig,
    rate_limiter: RateLimiter,
    client: genai.Client,
) -> dict[str, Any]:
    """Run `audit_runs_per_image` audits per file and set `compliant` from majority / mean tie-break."""
    n = config.audit_runs_per_image
    audit_runs: list[dict[str, Any]] = []
    for i in range(n):
        run_cfg = replace(config, seed=config.seed + i)
        try:
            single = generate_audit_single(
                image_path,
                run_cfg,
                rate_limiter,
                client,
                run_index=i,
                run_label=f"({i + 1}/{n})",
            )
            audit_runs.append(single)
        except Exception as exc:
            audit_runs.append(
                {
                    "filename": image_path.name,
                    "audit_run_index": i,
                    "error": str(exc),
                    "raw_error_type": type(exc).__name__,
                }
            )

    ok = [r for r in audit_runs if "error" not in r]
    if not ok:
        return {
            "filename": image_path.name,
            "error": "All audit runs failed for this image.",
            "raw_error_type": "AllRunsFailed",
            "audit_runs": audit_runs,
        }

    compliant_decision, agg_meta = _aggregate_compliant_from_runs(ok)
    rep = _representative_audit_run(ok, compliant_decision)
    models_order: list[str] = []
    seen_m: set[str] = set()
    for r in ok:
        m = r.get("gemini_model")
        if isinstance(m, str) and m not in seen_m:
            seen_m.add(m)
            models_order.append(m)

    merged: dict[str, Any] = {
        "filename": image_path.name,
        "compliant": compliant_decision,
        "compliant_aggregation": {
            "runs_requested": n,
            "runs_succeeded": len(ok),
            "runs_failed": n - len(ok),
            **agg_meta,
        },
        "violations": rep.get("violations"),
        "charts_detected": rep.get("charts_detected"),
        "final_explanation": rep.get("final_explanation"),
        "gemini_model": rep.get("gemini_model"),
        "gemini_models_used": models_order,
        "audit_runs": audit_runs,
    }
    return merged


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_rule_key(rule: Any) -> str:
    if not rule or not isinstance(rule, str):
        return "unknown"
    r = rule.strip()
    if "|" in r:
        r = r.split("|", maxsplit=1)[0].strip()
    return r or "unknown"


def write_compliance_charts(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plots summarizing the audit JSON (compliant, violations, charts_detected)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parsed = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    compliant_n = sum(1 for r in parsed if r.get("compliant") is True)
    noncompliant_n = sum(1 for r in parsed if r.get("compliant") is False)
    ambiguous_n = len(parsed) - compliant_n - noncompliant_n

    rule_counts: Counter[str] = Counter()
    confidences: list[float] = []
    low_conf_violations = 0
    chart_type_counts: Counter[str] = Counter()
    starts_at_zero_counts: Counter[str] = Counter()

    for r in parsed:
        violations = r.get("violations")
        if isinstance(violations, list):
            for v in violations:
                if not isinstance(v, dict):
                    continue
                rule_counts[_normalize_rule_key(v.get("rule"))] += 1
                c = _safe_float(v.get("confidence"))
                if c is not None:
                    confidences.append(c)
                if v.get("low_confidence") is True:
                    low_conf_violations += 1

        charts = r.get("charts_detected")
        if isinstance(charts, list):
            for ch in charts:
                if not isinstance(ch, dict):
                    continue
                chart_type_counts[str(ch.get("type") or "unknown")] += 1
                s = ch.get("starts_at_zero")
                if s is True:
                    starts_at_zero_counts["true"] += 1
                elif s is False:
                    starts_at_zero_counts["false"] += 1
                else:
                    starts_at_zero_counts["unknown"] += 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("IBCS audit — summary from model JSON", fontsize=14, fontweight="bold")

    # 1) Compliance outcome (per dashboard file)
    ax_c = axes[0, 0]
    labels = ["Compliant", "Non-compliant", "Unclear / missing flag", "API / parse error"]
    counts = [compliant_n, noncompliant_n, ambiguous_n, len(failed)]
    colors = ["#41ab5d", "#cb181d", "#fdae61", "#969696"]
    if sum(counts) > 0:
        x = range(len(labels))
        ax_c.bar(x, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax_c.set_xticks(list(x))
        ax_c.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax_c.set_ylabel("Dashboards")
        ax_c.set_title("Compliance outcome (by file)")
        for i, v in enumerate(counts):
            if v > 0:
                ax_c.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    else:
        ax_c.set_axis_off()

    # 2) Violations by rule (aggregated across all dashboards)
    ax_r = axes[0, 1]
    if rule_counts:
        items = rule_counts.most_common()
        rules = [k[:32] + ("…" if len(k) > 32 else "") for k, _ in items]
        vals = [v for _, v in items]
        y_pos = range(len(rules))
        ax_r.barh(list(y_pos), vals, color="#8856a7", edgecolor="white", linewidth=0.5)
        ax_r.set_yticks(list(y_pos))
        ax_r.set_yticklabels(rules, fontsize=8)
        ax_r.invert_yaxis()
        ax_r.set_xlabel("Violation count")
        ax_r.set_title("Violations by rule field")
    else:
        ax_r.text(0.5, 0.5, "No violations in JSON", ha="center", va="center")
        ax_r.set_axis_off()

    # 3) Charts detected per dashboard
    ax_d = axes[1, 0]
    if parsed:
        names = [str(r.get("filename", "?")) for r in parsed]
        n_charts = [
            len(r["charts_detected"]) if isinstance(r.get("charts_detected"), list) else 0
            for r in parsed
        ]
        y_pos = range(len(names))
        ax_d.barh(list(y_pos), n_charts, color="#2c7fb8", edgecolor="white", linewidth=0.5)
        ax_d.set_yticks(list(y_pos))
        ax_d.set_yticklabels(names, fontsize=8)
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Count")
        ax_d.set_title("charts_detected entries per file")
    else:
        ax_d.text(0.5, 0.5, "No parsed results", ha="center", va="center")
        ax_d.set_axis_off()

    # 4) Violation confidence + chart types (twin textual stats in subtitle; histogram for confidences)
    ax_h = axes[1, 1]
    if confidences:
        ax_h.hist(confidences, bins=min(12, max(4, len(confidences))), color="#7fcdbb", edgecolor="white")
        ax_h.set_xlabel("confidence (violations)")
        ax_h.set_ylabel("Count")
        ax_h.set_title("Distribution of violation confidence")
        ax_h.axvline(0.6, color="#cb181d", linestyle="--", linewidth=1, label="0.6 threshold")
        ax_h.legend(loc="upper right", fontsize=8)
    elif chart_type_counts:
        types = list(chart_type_counts.keys())
        vals = [chart_type_counts[t] for t in types]
        ax_h.bar(range(len(types)), vals, color="#fdae61", edgecolor="white", linewidth=0.5)
        ax_h.set_xticks(range(len(types)))
        ax_h.set_xticklabels([t[:20] for t in types], rotation=25, ha="right", fontsize=8)
        ax_h.set_ylabel("Count")
        ax_h.set_title("Chart types (charts_detected.type)")
    else:
        ax_h.text(0.5, 0.5, "No violation scores or chart types", ha="center", va="center")
        ax_h.set_axis_off()

    extra = (
        f"Files: {len(results)} · Parsed: {len(parsed)} · "
        f"Violation rows: {sum(rule_counts.values())} · "
        f"low_confidence violations: {low_conf_violations} · "
        f"starts_at_zero: {dict(starts_at_zero_counts)}"
    )
    fig.text(0.5, 0.02, extra, ha="center", fontsize=8, color="#333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def audit_dashboards(config: AuditConfig) -> list[dict[str, Any]]:
    set_reproducible_seed(config.seed)
    image_paths = get_image_paths(config.image_dir)
    rate_limiter = RateLimiter(
        min_interval_sec=config.min_interval_sec,
        max_requests_per_minute=config.max_requests_per_minute,
    )

    fb = config.vision_fallback_models or []
    fb_note = f" + {len(fb)} fallback(s)" if fb else ""
    try:
        import PIL  # noqa: F401

        _pillow = True
    except ImportError:
        _pillow = False
    resize_note = (
        f"Image resize: max edge {IMAGE_MAX_EDGE}px (GEMINI_IMAGE_MAX_EDGE; needs Pillow)."
        if _pillow and IMAGE_MAX_EDGE > 0
        else (
            "Install Pillow for faster runs: images are shrunk before request (see GEMINI_IMAGE_MAX_EDGE)."
            if not _pillow and IMAGE_MAX_EDGE > 0
            else "Image resize off (GEMINI_IMAGE_MAX_EDGE=0 or no Pillow)."
        )
    )
    client = genai.Client(api_key=config.api_key)
    pause_note = (
        f", +{config.post_request_pause_sec:g}s after each success"
        if config.post_request_pause_sec > 0
        else ""
    )
    print(
        f"Using Gemini API model {config.model!r}{fb_note} (set GEMINI_API_KEY). "
        f"max_output_tokens={config.max_output_tokens}. "
        f"HTTP timeout={config.http_timeout_sec:.0f}s. "
        f"Client throttle: ≥{config.min_interval_sec:g}s between API calls, "
        f"≤{config.max_requests_per_minute} RPM{pause_note}. "
        f"Confirm your project RPM/TPM/RPD in AI Studio and adjust GEMINI_MIN_INTERVAL_SEC / "
        f"GEMINI_MAX_RPM / GEMINI_POST_REQUEST_PAUSE_SEC if needed: "
        f"https://aistudio.google.com/rate-limit · "
        f"https://ai.google.dev/gemini-api/docs/rate-limits"
    )
    print(resize_note)
    sys.stdout.flush()
    print(
        "Note: tqdm stays at 0% until the first image finishes; vision requests often take several minutes."
    )
    sys.stdout.flush()
    print(
        f"Found {len(image_paths)} image(s). "
        f"{config.audit_runs_per_image} audit run(s) per image (GEMINI_AUDIT_RUNS). Starting audit..."
    )
    results: list[dict[str, Any]] = []

    with tqdm(
        image_paths,
        desc="Auditing dashboards",
        mininterval=0.5,
        miniters=1,
    ) as pbar:
        for image_path in pbar:
            try:
                short = image_path.name[:44] + ("…" if len(image_path.name) > 44 else "")
                pbar.set_postfix_str(short, refresh=True)
                results.append(generate_audit(image_path, config, rate_limiter, client))
            except Exception as exc:
                print(f"\nError processing {image_path.name}: {exc}")
                results.append(
                    {
                        "filename": image_path.name,
                        "error": str(exc),
                        "raw_error_type": type(exc).__name__,
                    }
                )

    return results


def save_results(results: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main() -> None:
    config = AuditConfig()
    results = audit_dashboards(config)
    save_results(results, config.output_file)
    print(f"\nAudit complete. Report saved to: {config.output_file.resolve()}")
    try:
        write_compliance_charts(results, config.compliance_charts_file)
        print(f"Compliance charts saved to: {config.compliance_charts_file.resolve()}")
    except ImportError:
        print(
            "matplotlib is not installed; skipping compliance charts. "
            "Install with: pip install matplotlib"
        )


if __name__ == "__main__":
    main()

"""LLM wrapper supporting two providers. The app picks one from the key you set:

- Groq   (GROQ_API_KEY,   starts with "gsk_")  -> Llama 3.3 70B, replies in ~1-2 s. Recommended.
- Gemini (GEMINI_API_KEY)                      -> Gemini Flash, thinking switched off.

Handles the failure modes the report asks about:
- invalid key            -> clear message, no retry
- model retired/renamed  -> falls back to the next model in the list
- rate limit / overload  -> waits and retries
- empty or broken JSON   -> raises BadOutput so the app can recover gracefully
"""

import json
import re
import time

GROQ_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-2.0-flash"]
PROVIDER_NAMES = {"groq": "Groq (Llama)", "gemini": "Google Gemini"}


class GeminiError(Exception):
    """The API could not be reached or refused the request. (Name kept for compatibility.)"""

    def __init__(self, user_message, detail=""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class BadOutput(Exception):
    """The model answered, but not with usable JSON."""


def parse_json(text):
    if not text:
        raise BadOutput("empty response")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise BadOutput(f"could not parse: {text[:200]}")


def detect_provider(api_key):
    return "groq" if api_key.startswith("gsk_") else "gemini"


class GeminiClient:
    """Despite the name, talks to Groq or Gemini depending on the key."""

    def __init__(self, api_key, preferred_model=None):
        self.provider = detect_provider(api_key)
        self.provider_name = PROVIDER_NAMES[self.provider]
        if self.provider == "groq":
            from groq import Groq

            self.client = Groq(api_key=api_key, max_retries=0)
            defaults = GROQ_MODELS
        else:
            from google import genai

            self.client = genai.Client(api_key=api_key)
            defaults = GEMINI_MODELS
        # Only honour a preferred model that belongs to this provider.
        is_gemini_name = bool(preferred_model) and preferred_model.startswith("gemini")
        models = [preferred_model] if preferred_model and is_gemini_name == (self.provider == "gemini") else []
        self.models = models + [m for m in defaults if m not in models]
        self.active_model = None
        self.thinking_fixed = set()

    def generate_json(self, system, prompt, temperature=0.7):
        return parse_json(self._generate(system, prompt, temperature))

    def _call(self, model, system, prompt, temperature):
        if self.provider == "groq":
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=30,
            )
            return resp.choices[0].message.content

        from google.genai import types

        kwargs = dict(system_instruction=system, temperature=temperature, response_mime_type="application/json")
        if model not in self.thinking_fixed:
            # Switch off Gemini's internal "thinking" step: it is the main source of delay.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = self.client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs)
        )
        return resp.text

    def _generate(self, system, prompt, temperature):
        order = ([self.active_model] if self.active_model else []) + [
            m for m in self.models if m != self.active_model
        ]
        key_name = "GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY"
        errors = []
        for model in order:
            for attempt in range(3):
                try:
                    text = self._call(model, system, prompt, temperature)
                    if not text:
                        raise BadOutput("empty response (possibly blocked by safety filters)")
                    self.active_model = model
                    return text
                except BadOutput as e:
                    errors.append(f"{model}: {e}")
                    time.sleep(0.5)
                    continue
                except Exception as e:  # SDKs raise several error classes; inspect the message
                    msg = str(e)
                    low = msg.lower()
                    errors.append(f"{model}: {msg[:200]}")
                    if (
                        "api key" in low or "api_key" in low or "unauthenticated" in low
                        or "permission_denied" in low or "401" in msg
                    ):
                        raise GeminiError(
                            f"{self.provider_name} rejected the API key. Check {key_name} in your Streamlit secrets.",
                            msg,
                        )
                    if self.provider == "gemini" and "thinking" in low and model not in self.thinking_fixed:
                        self.thinking_fixed.add(model)  # model can't disable thinking; retry without it
                        continue
                    if (
                        "404" in msg or "not found" in low or "not supported" in low
                        or "does not exist" in low or "decommissioned" in low
                    ):
                        break  # this model is unavailable; try the next one
                    if "json_validate_failed" in low or "failed to generate json" in low:
                        time.sleep(0.3)  # Groq occasionally fails JSON mode; just retry
                        continue
                    if "429" in msg or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
                        time.sleep(2 * (attempt + 1))
                        continue
                    time.sleep(1 * (attempt + 1))  # 5xx, timeouts, network blips
        raise GeminiError(
            f"Couldn't reach {self.provider_name} after several tries (likely the free-tier rate limit). "
            "Wait about 30 seconds and send your answer again.",
            "\n".join(errors),
        )

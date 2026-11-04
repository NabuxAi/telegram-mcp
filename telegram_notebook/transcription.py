from __future__ import annotations

from pathlib import Path

from .media import normalize_audio_input
from .provider_http import gemini_transcribe_audio


class TranscriptionService:
    def __init__(self, *, provider: str, api_key: str | None, model: str, base_url: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self._whisper = None

    @property
    def enabled(self) -> bool:
        # Local Whisper needs no API key; cloud providers do.
        if self.provider in ("local", "whisper"):
            return True
        return self.api_key is not None

    def _get_client(self):
        if not self.api_key:
            return None
        if self.client is None:
            if self.provider != "gemini":
                from openai import OpenAI

                # base_url, not just the key: NabuGate speaks the OpenAI wire
                # protocol, so without this a gateway-provider request would be
                # sent to api.openai.com carrying a gateway token.
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        return self.client

    def _get_whisper(self):
        """Lazily load a local faster-whisper model (heavy, opt-in dependency)."""
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Local transcription needs faster-whisper. Install it with: "
                    "pip install '.[local]'  (or: pip install faster-whisper)"
                ) from exc
            # device/compute_type='auto' lets it use a GPU when present, else CPU.
            self._whisper = WhisperModel(self.model or "base", device="auto", compute_type="auto")
        return self._whisper

    def _transcribe_local(self, segment: Path) -> str:
        model = self._get_whisper()
        segments, _info = model.transcribe(str(segment))
        return " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip()).strip()

    def transcribe_media(self, source_path: Path, work_dir: Path) -> str:
        if self.provider not in ("local", "whisper") and not self.api_key:
            raise RuntimeError(f"API key for provider '{self.provider}' is required for transcription")

        segments = normalize_audio_input(source_path, work_dir)
        texts: list[str] = []
        for segment in segments:
            if self.provider in ("local", "whisper"):
                text = self._transcribe_local(segment)
                if text:
                    texts.append(text)
            elif self.provider == "gemini":
                text = gemini_transcribe_audio(
                    api_key=self.api_key,
                    model=self.model,
                    audio_path=segment,
                )
                if text:
                    texts.append(text)
            else:
                client = self._get_client()
                if not client:
                    raise RuntimeError("OPENAI_API_KEY is required for transcription")
                with segment.open("rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="text",
                    )
                if hasattr(response, "text"):
                    texts.append(response.text)
                else:
                    texts.append(str(response))
        return "\n".join(part.strip() for part in texts if part.strip()).strip()

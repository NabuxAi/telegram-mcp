from __future__ import annotations

import base64
import json
import logging
import ssl
import subprocess
from pathlib import Path
from urllib import request

import certifi

from .media import guess_mime_type

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://aiplatform.googleapis.com/v1/publishers/google"


def get_gcloud_access_token() -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        logger.exception("Error getting gcloud access token")
        return None


def _json_request(
    *,
    url: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    use_gcloud_auth: bool = False,
) -> dict[str, object] | list[dict[str, object]]:
    raw = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    if use_gcloud_auth:
        token = get_gcloud_access_token()
        if token:
            req_headers["Authorization"] = f"Bearer {token}"
            
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
    
    req = request.Request(url, data=raw, headers=req_headers)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    try:
        with request.urlopen(req, timeout=90, context=ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        logger.exception("Vertex AI request error to %s", url)
        raise


def list_gemini_models(
    *,
    api_key: str,
    capability: str | None = None,
) -> list[dict[str, object]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    output: list[dict[str, object]] = []

    for model in client.models.list():
        model_id = str(getattr(model, "name", "") or "")
        if model_id.startswith("models/"):
            model_id = model_id.split("/", 1)[1]
        display_name = str(getattr(model, "display_name", "") or model_id)
        description = str(getattr(model, "description", "") or "")
        supported_actions = " ".join(str(v) for v in (getattr(model, "supported_actions", None) or []))
        haystack = f"{model_id} {display_name} {description} {supported_actions}".lower()

        if capability == "generateContent" and "gemini" not in haystack:
            continue
        if capability == "embedContent" and "embed" not in haystack:
            continue

        output.append(
            {
                "id": model_id,
                "display_name": display_name,
                "provider": "gemini",
            }
        )

    return sorted(output, key=lambda item: str(item["id"]))


def list_openai_models(
    *,
    api_key: str,
    capability: str | None = None,
) -> list[dict[str, object]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    output: list[dict[str, object]] = []

    for model in client.models.list():
        model_id = getattr(model, "id", "")
        if not model_id:
            continue
        lowered = model_id.lower()
        if capability == "transcription" and "transcribe" not in lowered:
            continue
        if capability == "embedding" and "embedding" not in lowered:
            continue
        output.append(
            {
                "id": model_id,
                "display_name": model_id,
                "provider": "openai",
            }
        )

    return sorted(output, key=lambda item: str(item["id"]))


def vertex_ai_search(
    *,
    api_key: str | None = None,
    project_id: str,
    region: str,
    index_endpoint_id: str,
    deployed_index_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, object]]:
    # https://cloud.google.com/vertex-ai/docs/vector-search/query-index-public-endpoint
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}:findNeighbors?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}:findNeighbors"
        use_gcloud = True
    
    payload = {
        "deployed_index_id": deployed_index_id,
        "queries": [
            {
                "datapoint": {"feature_vector": query_embedding},
                "neighbor_count": top_k
            }
        ]
    }
    
    data = _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)
    results = []
    if isinstance(data, dict):
        # Vertex AI returns a list of nearestNeighbors for each query
        nearest_neighbors_list = data.get("nearestNeighbors", [])
        if nearest_neighbors_list:
            neighbors = nearest_neighbors_list[0].get("neighbors", [])
            for n in neighbors:
                results.append({
                    "id": n.get("datapoint", {}).get("datapointId"),
                    "distance": n.get("distance")
                })
    return results


def vertex_ai_upsert(
    *,
    api_key: str | None = None,
    project_id: str,
    region: str,
    index_id: str,
    datapoints: list[dict[str, object]],
) -> None:
    # https://cloud.google.com/vertex-ai/docs/vector-search/upsert-datapoints
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexes/{index_id}:upsertDatapoints?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexes/{index_id}:upsertDatapoints"
        use_gcloud = True
    
    payload = {
        "datapoints": datapoints
    }
    
    _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)


def gemini_embed_text(
    *,
    api_key: str | None = None,
    model: str,
    text: str,
    task_type: str | None = None,
    project_id: str | None = None,
    region: str = "us-central1",
) -> list[float] | None:
    if api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.EmbedContentConfig(task_type=task_type) if task_type else None
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=config,
        )
        embeddings = getattr(response, "embeddings", None) or []
        if embeddings:
            values = getattr(embeddings[0], "values", None)
            if values:
                return list(values)
        return None

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:predict"
    payload = {
        "instances": [{"content": text}],
    }
    data = _json_request(url=url, payload=payload, use_gcloud_auth=True)
    if isinstance(data, dict):
        predictions = data.get("predictions", [])
        if predictions:
            return predictions[0].get("embeddings", {}).get("values", [])
    return None


def gemini_transcribe_audio(
    *,
    api_key: str | None = None,
    model: str,
    audio_path: Path,
    project_id: str | None = None,
    region: str = "us-central1",
) -> str:
    if "flash" not in model:
        model = "gemini-2.5-flash-lite"

    mime_type = guess_mime_type(audio_path) or "audio/mpeg"

    if api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                "Generate a verbatim transcript of the spoken audio. Return only the transcript text.",
                types.Part.from_bytes(data=audio_path.read_bytes(), mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else ""

    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Generate a verbatim transcript of the spoken audio. Return only the transcript text."},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded_audio,
                        }
                    },
                ]
            }
        ]
    }

    data = _json_request(url=url, payload=payload, use_gcloud_auth=True)

    full_text = []
    if isinstance(data, list):
        for chunk in data:
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        full_text.append(p["text"])
    elif isinstance(data, dict):
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    full_text.append(p["text"])
                
    return "".join(full_text).strip()


_EXTRACT_PROMPT = (
    "Extract all readable text from this file (a document, scan, or photo). "
    "For images, perform OCR. Return only the extracted text, preserving order. "
    "If there is no readable text, return an empty response."
)


def gemini_extract_document(
    *,
    api_key: str | None = None,
    model: str,
    file_path: Path,
    project_id: str | None = None,
    region: str = "us-central1",
) -> str:
    """OCR / text extraction for an image or PDF via Gemini multimodal."""
    if "flash" not in model:
        model = "gemini-2.5-flash-lite"
    mime_type = guess_mime_type(file_path) or "application/octet-stream"

    if api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                _EXTRACT_PROMPT,
                types.Part.from_bytes(data=file_path.read_bytes(), mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else ""

    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _EXTRACT_PROMPT},
                    {"inlineData": {"mimeType": mime_type, "data": encoded}},
                ],
            }
        ]
    }
    data = _json_request(url=url, payload=payload, use_gcloud_auth=True)
    full_text = []
    chunks = data if isinstance(data, list) else [data]
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        candidates = chunk.get("candidates", [])
        if candidates:
            for p in candidates[0].get("content", {}).get("parts", []):
                if "text" in p:
                    full_text.append(p["text"])
    return "".join(full_text).strip()


def gemini_generate_content(
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash-lite",
    prompt: str,
    project_id: str | None = None,
    region: str = "us-central1",
) -> str:
    if api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40,
            ),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else ""

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40,
        }
    }

    data = _json_request(url=url, payload=payload, use_gcloud_auth=True)

    full_text = []
    if isinstance(data, list):
        for chunk in data:
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        full_text.append(p["text"])
    elif isinstance(data, dict):
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    full_text.append(p["text"])
                
    return "".join(full_text).strip()

"""RunPod serverless worker — FLUX.1-Kontext-dev (image editing / img2img by instruction).

입력 이미지 + 프롬프트(편집 지시) → 편집된 이미지. diffusers FluxKontextPipeline 사용.
모델은 gated → HF_TOKEN 필요(라이선스 수락한 계정 토큰). 대용량(~24GB)이라 HF 캐시를
네트워크 볼륨(/runpod-volume)에 두는 것을 강력 권장(콜드스타트마다 재다운로드 방지).

Input (event["input"]):
  prompt            (str, required)  편집 지시. 예: "make the background white", "add a hat"
  image             (str, required)  입력 이미지 — data URI / base64 / http(s) URL 중 하나
                                     (image_url / image_base64 별칭도 허용)
  guidance_scale    (float, opt=2.5)
  num_inference_steps (int, opt)     기본 파이프라인 값 사용
  seed              (int, opt)       재현용
Output:
  { "image": "data:image/png;base64,...." }  또는 { "error": "..." }
"""

import os
import io
import base64
import traceback

import torch
import requests
from PIL import Image
import runpod

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "0") == "1"   # 24GB GPU면 1 권장(오프로드), 40GB+면 0
DTYPE = torch.bfloat16

_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import FluxKontextPipeline
        print(f"[flux-kontext] 파이프라인 로드: {MODEL_ID} (offload={CPU_OFFLOAD})", flush=True)
        _pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, token=HF_TOKEN)
        if CPU_OFFLOAD:
            _pipe.enable_model_cpu_offload()      # VRAM 절약(24GB GPU 대응), 약간 느림
        else:
            _pipe.to("cuda")
        print("[flux-kontext] 로드 완료", flush=True)
    return _pipe


def _load_image(spec):
    if not isinstance(spec, str) or not spec:
        raise ValueError("image is required")
    if spec.startswith("http://") or spec.startswith("https://"):
        r = requests.get(spec, timeout=60)
        r.raise_for_status()
        data = r.content
    else:
        if spec.startswith("data:"):
            spec = spec.split(",", 1)[1]
        data = base64.b64decode(spec)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(event):
    try:
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        image_spec = inp.get("image") or inp.get("image_url") or inp.get("image_base64")
        if not prompt:
            return {"error": "prompt is required"}
        if not image_spec:
            return {"error": "image is required (data URI / base64 / URL)"}

        img = _load_image(image_spec)
        pipe = get_pipe()

        kwargs = {
            "image": img,
            "prompt": str(prompt),
            "guidance_scale": float(inp.get("guidance_scale", 2.5)),
        }
        if inp.get("num_inference_steps"):
            kwargs["num_inference_steps"] = int(inp["num_inference_steps"])
        if inp.get("seed") is not None:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(inp["seed"]))

        out = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image": "data:image/png;base64," + b64}
    except Exception as e:
        print("[flux-kontext] 오류:", e, traceback.format_exc(), flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

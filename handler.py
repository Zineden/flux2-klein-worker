"""RunPod serverless worker — FLUX.2-klein-4B (Apache-2.0, 상업이용 가능, 비게이트).

이미지 편집(입력 이미지 + 프롬프트 → 편집 이미지) 및 텍스트→이미지 모두 지원.
diffusers Flux2KleinPipeline 사용. 모델이 gated 아님 → HF 토큰 불필요.
~13GB VRAM(16GB GPU면 충분), 4-step 증류 모델이라 빠름. 멀티 레퍼런스 이미지 지원.

Input (event["input"]):
  prompt              (str, required)  편집 지시 / 생성 프롬프트
  image | image_url | image_base64  (str, opt)  입력/레퍼런스 이미지 1장(편집 모드)
  images              (array, opt)  멀티 레퍼런스(여러 장). image 대신 사용 가능
  guidance_scale      (float, opt=2.5)
  num_inference_steps (int, opt=4)
  height, width       (int, opt)   미지정 시 파이프라인 기본값
  seed                (int, opt)
Output:
  { "image": "data:image/png;base64,..." }  또는 { "error": "..." }
"""

import os
import io
import base64
import traceback

import torch
import requests
from PIL import Image
import runpod

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")  # 비게이트라 없어도 됨
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "1") == "1"   # 기본 ON(VRAM 절약, 16GB GPU 대응)
DTYPE = torch.bfloat16

_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import Flux2KleinPipeline
        print(f"[flux2-klein] 파이프라인 로드: {MODEL_ID} (offload={CPU_OFFLOAD})", flush=True)
        _pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, token=HF_TOKEN)
        if CPU_OFFLOAD:
            _pipe.enable_model_cpu_offload()
        else:
            _pipe.to("cuda")
        print("[flux2-klein] 로드 완료", flush=True)
    return _pipe


def _load_image(spec):
    if not isinstance(spec, str) or not spec:
        raise ValueError("빈 이미지 값")
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
        if not prompt:
            return {"error": "prompt is required"}

        # 레퍼런스/입력 이미지 수집(편집 모드). 없으면 텍스트→이미지.
        specs = []
        if inp.get("images"):
            specs = inp["images"] if isinstance(inp["images"], list) else [inp["images"]]
        else:
            one = inp.get("image") or inp.get("image_url") or inp.get("image_base64")
            if one:
                specs = [one]
        imgs = [_load_image(s) for s in specs]

        pipe = get_pipe()
        kwargs = {
            "prompt": str(prompt),
            "guidance_scale": float(inp.get("guidance_scale", 2.5)),
            "num_inference_steps": int(inp.get("num_inference_steps", 4)),
        }
        if imgs:
            kwargs["image"] = imgs if len(imgs) > 1 else imgs[0]   # 멀티면 리스트, 1장이면 단일
        if inp.get("height"):
            kwargs["height"] = int(inp["height"])
        if inp.get("width"):
            kwargs["width"] = int(inp["width"])
        if inp.get("seed") is not None:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(inp["seed"]))

        out = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image": "data:image/png;base64," + b64}
    except Exception as e:
        print("[flux2-klein] 오류:", e, traceback.format_exc(), flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

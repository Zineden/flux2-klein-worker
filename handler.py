"""RunPod serverless worker — FLUX.2-klein-4B (Apache-2.0, 상업이용 가능, 비게이트).

이미지 편집(입력 이미지 + 프롬프트 → 편집 이미지) 및 텍스트→이미지 모두 지원.
diffusers Flux2KleinPipeline 사용. 모델이 gated 아님 → HF 토큰 불필요.
~13GB VRAM(16GB GPU면 충분), 4-step 증류 모델이라 빠름. 멀티 레퍼런스 이미지 지원.

이미지 소스(alicdn 등)는 워커가 직접 받는다 — 브라우저 User-Agent 헤더 필수(없으면 alicdn 420).
결과는 Cloudflare R2에 업로드하고 공개 URL을 반환한다(bg/qwen 워커와 동일 패턴).
R2 미설정 시 base64(data URI)로 폴백.

Input (event["input"]):
  prompt              (str, required)  편집 지시 / 생성 프롬프트
  image | image_url | image_base64  (str, opt)  입력/레퍼런스 이미지 1장(편집 모드)
  images              (array, opt)  멀티 레퍼런스(여러 장). image 대신 사용 가능
  guidance_scale      (float, opt=2.5)
  num_inference_steps (int, opt=4)
  height, width       (int, opt)   미지정 시 파이프라인 기본값
  seed                (int, opt)
Output:
  { "image_url": "https://<r2>/klein/....png" }  (R2 설정 시)
  { "image": "data:image/png;base64,..." }       (R2 미설정 폴백)
  { "error": "..." }
"""

import os
import io
import time
import base64
import hashlib
import threading
import traceback
import urllib.error
import urllib.request

import torch
from PIL import Image
import runpod

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")  # 비게이트라 없어도 됨
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "1") == "1"   # 기본 ON(VRAM 절약, 16GB GPU 대응)
DTYPE = torch.bfloat16
DOWNLOAD_TIMEOUT = int(os.environ.get("KLEIN_DOWNLOAD_TIMEOUT", "60"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("KLEIN_MAX_DOWNLOAD_BYTES", str(25 * 1024 * 1024)))

# --- R2 (선택): 출력 저장. bg/qwen 워커와 동일한 env 이름을 쓴다(같은 값 복사). ---
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE")
R2_PREFIX = os.environ.get("R2_PREFIX", "klein")

_pipe = None
_s3 = None
_s3_lock = threading.Lock()


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


# --- R2 helpers (bg 워커와 동일 패턴) ---
def r2_enabled():
    return all([R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY, R2_PUBLIC_BASE])


def get_s3():
    global _s3
    if _s3 is None:
        with _s3_lock:
            if _s3 is None:
                import boto3
                _s3 = boto3.client(
                    "s3",
                    endpoint_url=R2_ENDPOINT,
                    aws_access_key_id=R2_ACCESS_KEY,
                    aws_secret_access_key=R2_SECRET_KEY,
                )
    return _s3


def public_url(key):
    return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"


def r2_head(key):
    try:
        get_s3().head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except Exception:
        return None


def _fetch_bytes(url):
    req = urllib.request.Request(
        url,
        headers={
            # alicdn 핫링크/레이트 차단(420) 회피 — 브라우저 UA 필수.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://detail.1688.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                data = resp.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError("image exceeds size limit")
            return data
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (408, 429):
                raise
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < 2:
            time.sleep(0.4 * (attempt + 1))
    raise last_err if last_err else RuntimeError("download failed")


def _source_bytes(spec):
    """URL(alicdn 등) 또는 base64/data-uri → 원본 바이트."""
    if not isinstance(spec, str) or not spec:
        raise ValueError("빈 이미지 값")
    if spec.startswith("http://") or spec.startswith("https://"):
        return _fetch_bytes(spec)
    if spec.startswith("data:"):
        spec = spec.split(",", 1)[1]
    return base64.b64decode(spec)


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

        steps = int(inp.get("num_inference_steps", 4))
        guidance = float(inp.get("guidance_scale", 2.5))

        src_bytes = [_source_bytes(s) for s in specs]
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in src_bytes]

        # 콘텐츠 주소(캐시/디둡): sha256(원본들 + 프롬프트 + 파라미터). seed는 제외(반복 호출 디둡).
        key = None
        if r2_enabled():
            h = hashlib.sha256()
            for b in src_bytes:
                h.update(b)
            h.update(("|" + str(prompt) + "|" + str(steps) + "|" + str(guidance)).encode("utf-8"))
            key = f"{R2_PREFIX.strip('/')}/{h.hexdigest()}.png"
            if r2_head(key) is not None:
                print(f"[flux2-klein] R2 캐시 히트: {key}", flush=True)
                return {"image_url": public_url(key), "cached": True}

        pipe = get_pipe()
        kwargs = {
            "prompt": str(prompt),
            "guidance_scale": guidance,
            "num_inference_steps": steps,
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
        png = buf.getvalue()

        # R2 업로드 → 공개 URL 반환(bg/qwen 워커와 동일). 미설정 시 base64 폴백.
        if r2_enabled():
            try:
                get_s3().put_object(
                    Bucket=R2_BUCKET,
                    Key=key,
                    Body=png,
                    ContentType="image/png",
                    CacheControl="public, max-age=31536000, immutable",
                )
                return {"image_url": public_url(key)}
            except Exception as e:
                print("[flux2-klein] R2 업로드 실패, base64 폴백:", e, flush=True)

        b64 = base64.b64encode(png).decode("ascii")
        return {"image": "data:image/png;base64," + b64}
    except Exception as e:
        print("[flux2-klein] 오류:", e, traceback.format_exc(), flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

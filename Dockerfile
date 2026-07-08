# FLUX.2-klein-4B RunPod serverless worker (diffusers) — 모델을 이미지에 굽는다(네트워크 볼륨 불필요).
# torch 2.9(cuda12.8): diffusers-git의 FLUX.2 autoencoder가 등록하는 FlashAttention-3
# custom op은 PEP-604 유니온 주석(`float | None`)을 쓰는데, torch 2.4의 infer_schema는
# 이를 파싱하지 못해 임포트가 실패했다. torch 2.9는 지원. RunPod 호스트도 CUDA 12.8.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # 모델을 빌드 시점에 이미지 안(HF 캐시)에 굽는다 → 네트워크 볼륨 불필요,
    # 런타임 다운로드 없음(콜드스타트마다 재다운로드/디스크부족 문제 제거).
    HF_HOME=/opt/hf

# git 필요: requirements 의 diffusers 를 git 저장소에서 설치(Flux2KleinPipeline 신규 클래스).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# hf_xet(huggingface_hub 1.x의 Xet 백엔드) 제거 — 빌드 시 대용량 파일 다운로드가 0%에서
# 멈추는 원인. HF_HUB_DISABLE_XET 플래그는 1.x에서 듣지 않으므로 패키지 자체를 제거해
# 일반 HTTPS 다운로더로 폴백시킨다(안정적). non-gated 모델이라 토큰 없이도 받는다.
RUN pip uninstall -y hf_xet || true

# 모델(~13GB)을 빌드 시점에 이미지로 다운로드(굽기). 런타임엔 이미 존재하므로 재다운로드 없음.
ARG MODEL_ID=black-forest-labs/FLUX.2-klein-4B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', max_workers=4)"

COPY handler.py .

# 상대경로 handler.py — RunPod의 정적 핸들러 검사(CMD 파싱)와 호환.
CMD ["python", "-u", "handler.py"]

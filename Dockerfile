# FLUX.2-klein-4B RunPod serverless worker (diffusers) — 모델을 이미지에 굽는다(네트워크 볼륨 불필요).
# torch 2.9(cuda12.8): diffusers-git의 FLUX.2 autoencoder가 등록하는 FlashAttention-3
# custom op은 PEP-604 유니온 주석(`float | None`)을 쓰는데, torch 2.4의 infer_schema는
# 이를 파싱하지 못해 임포트가 실패했다. torch 2.9는 지원. RunPod 호스트도 CUDA 12.8.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # 모델을 빌드 시점에 이미지 안(HF 캐시)에 굽는다 → 네트워크 볼륨 불필요,
    # 런타임 다운로드 없음(콜드스타트마다 재다운로드/디스크부족 문제 제거).
    HF_HOME=/opt/hf \
    # hf_transfer(Rust 가속 다운로더)로 대용량 모델을 빠르고 안정적으로 받는다
    # (일반 HTTPS는 8GB transformer에서 매우 느림/멈춤).
    HF_HUB_ENABLE_HF_TRANSFER=1

# git 필요: requirements 의 diffusers 를 git 저장소에서 설치(Flux2KleinPipeline 신규 클래스).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .
# requirements가 huggingface_hub 0.x + hf_transfer를 고정 설치한다(Xet 없음, 고속 다운로더 유효).
RUN pip install --no-cache-dir -r requirements.txt

# 모델(~13GB)을 빌드 시점에 이미지로 굽는다. huggingface_hub 0.x + hf_transfer로 고속 다운로드
# (HF_HUB_ENABLE_HF_TRANSFER=1). 런타임 재다운로드 없음.
ARG MODEL_ID=black-forest-labs/FLUX.2-klein-4B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

COPY handler.py .

# 상대경로 handler.py — RunPod의 정적 핸들러 검사(CMD 파싱)와 호환.
CMD ["python", "-u", "handler.py"]

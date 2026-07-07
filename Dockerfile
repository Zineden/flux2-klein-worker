# FLUX.2-klein-4B RunPod serverless worker (diffusers)
# torch/cuda 포함 베이스 → pip 설치만 하므로 빌드가 가볍다(모델은 런타임에 HF에서 다운로드).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # HF 캐시를 네트워크 볼륨에 두어 콜드스타트마다 모델(~13GB) 재다운로드 방지.
    # RunPod 엔드포인트에 네트워크 볼륨을 붙이면 /runpod-volume 에 마운트됨.
    HF_HOME=/runpod-volume/huggingface

# git 필요: requirements 의 diffusers 를 git 저장소에서 설치(Flux2KleinPipeline 신규 클래스).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir hf_transfer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 상대경로 handler.py — RunPod의 정적 핸들러 검사(CMD 파싱)와 호환.
CMD ["python", "-u", "handler.py"]

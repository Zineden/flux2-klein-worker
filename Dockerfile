# FLUX.2-klein-4B RunPod serverless worker (diffusers)
# 모델은 런타임에 네트워크 볼륨(/runpod-volume)으로 1회 다운로드한다(bg/qwen 워커와 동일 패턴).
# 빌드 시 모델을 굽지 않는다 → 빌드가 가볍고 빠르며(코드 변경 재빌드도 빠름), 15GB 이미지를
# 워커 스케일업마다 다시 받는 비용도 없다. ※ 엔드포인트에 네트워크 볼륨(~25GB)을 붙일 것.
#
# torch 2.9(cuda12.8): diffusers-git의 FLUX.2 autoencoder가 등록하는 FlashAttention-3
# custom op은 PEP-604 유니온 주석(`float | None`)을 쓰는데, torch 2.4의 infer_schema는
# 이를 파싱하지 못해 임포트가 실패했다. torch 2.9는 지원. RunPod 호스트도 CUDA 12.8.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # HF 캐시를 네트워크 볼륨에 두어 콜드스타트마다 모델(~13GB) 재다운로드 방지.
    # RunPod 엔드포인트에 네트워크 볼륨을 붙이면 /runpod-volume 에 마운트됨.
    HF_HOME=/runpod-volume/huggingface

# git 필요: requirements 의 diffusers 를 git 저장소에서 설치(Flux2KleinPipeline 신규 클래스).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# hf_xet(huggingface_hub 1.x의 Xet 백엔드)는 대용량 파일 다운로드가 0%에서 멈추는 사례가 있고
# HF_HUB_DISABLE_XET 플래그도 1.x에선 잘 듣지 않는다 → 제거해 일반 HTTPS 다운로드로 폴백(안정적).
# non-gated 모델이라 토큰 없이도 받는다(속도만 조금 느림, 볼륨에 1회 캐시되므로 무방).
RUN pip uninstall -y hf_xet || true

COPY handler.py .

# 상대경로 handler.py — RunPod의 정적 핸들러 검사(CMD 파싱)와 호환.
CMD ["python", "-u", "handler.py"]

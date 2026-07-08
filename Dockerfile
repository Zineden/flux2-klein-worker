# FLUX.2-klein-4B RunPod serverless worker (diffusers) — RunPod Model Caching 사용.
# 모델을 이미지에 굽지 않는다 → 작은 이미지(빠른 워커 프로비저닝). RunPod가 HF 모델을
# 호스트에 미리 캐시(/runpod-volume/huggingface-cache)하므로 콜드스타트가 초 단위로 줄고,
# 다운로드 중에는 과금되지 않는다(=qwen 워커가 빠른 이유). 볼륨을 수동으로 붙일 필요 없음.
#
# ※ 엔드포인트 설정(콘솔)에서 Model 필드 = black-forest-labs/FLUX.2-klein-4B 로 캐싱 활성화할 것.
#
# torch 2.9(cuda12.8): FLUX.2 autoencoder의 FlashAttention-3 custom op(PEP-604 유니온 주석)을
# torch 2.4 infer_schema가 파싱 못해 실패 → 2.9 필요. RunPod 호스트도 CUDA 12.8.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # RunPod Model Caching 경로 → from_pretrained가 여기 미리 캐시된 모델을 읽는다(런타임 다운로드 없음).
    HF_HOME=/runpod-volume/huggingface-cache \
    # 모델은 RunPod이 워커 시작 전에 캐시 완료를 보장하므로, 오프라인 읽기로 네트워크 HEAD/다운로드 제거.
    HF_HUB_OFFLINE=1

# git 필요: requirements 의 diffusers 를 git 저장소에서 설치(Flux2KleinPipeline 신규 클래스).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 상대경로 handler.py — RunPod의 정적 핸들러 검사(CMD 파싱)와 호환.
CMD ["python", "-u", "handler.py"]

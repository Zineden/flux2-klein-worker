# FLUX.2-klein RunPod worker on wlsdml1114/multitalk-base (fleet-cached base → 콜드스타트 단축 시도).
# multitalk-base:1.7 = torch 2.7.0+cu128, python3.10, cuda12.8 (qwen/flux-kontext 워커와 동일 베이스).
# FLUX.2 diffusers는 이전에 torch 2.9를 요구했다(torch 2.4는 FlashAttention-3 custom op의
# PEP-604 유니온 주석을 infer_schema가 파싱 못해 실패). torch 2.7에서 되는지 빌드 시 임포트로 검증한다.
#   → 아래 RUN 임포트 체크가 통과하면 torch 2.7 OK(빌드 성공/배포), 실패하면 빌드가 여기서 멈춘다.
FROM wlsdml1114/multitalk-base:1.7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # RunPod Model Caching 경로(콘솔 Model 필드로 미리 캐시) → from_pretrained가 오프라인으로 읽음.
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_OFFLINE=1

WORKDIR /app

# torch는 재설치하지 않는다(베이스의 2.7.0 유지). diffusers(git)·transformers(Qwen3 TE용 4.5x)·boto3 등만 추가.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ▼ 핵심 검증: torch 2.7에서 Flux2KleinPipeline 임포트가 되는지 빌드 시점에 확인.
#   (임포트가 autoencoder_kl_flux2 → attention_dispatch의 FA3 custom op 등록을 트리거함)
RUN python -c "import torch; print('TORCH', torch.__version__); from diffusers import Flux2KleinPipeline; print('FLUX2-KLEIN IMPORT OK')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]

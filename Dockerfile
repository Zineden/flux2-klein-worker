# FLUX.1-Kontext-dev RunPod serverless worker (diffusers)
# torch/cuda 포함 베이스 → pip 설치만 하므로 빌드가 가볍다(모델은 런타임에 HF에서 다운로드).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # HF 캐시를 네트워크 볼륨에 두어 콜드스타트마다 24GB 재다운로드 방지.
    # RunPod 엔드포인트에 네트워크 볼륨을 붙이면 /runpod-volume 에 마운트됨.
    HF_HOME=/runpod-volume/huggingface

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir hf_transfer

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]

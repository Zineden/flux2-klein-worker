# flux-kontext-worker

RunPod serverless worker for **FLUX.1-Kontext-dev** — instruction-based image editing
(input image + text prompt → edited image), via diffusers `FluxKontextPipeline`.

## API

**Input** (`event["input"]`):

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | edit instruction, e.g. `"make the background white"` |
| `image` | string | ✅ | — | input image: data URI / base64 / http(s) URL (aliases: `image_url`, `image_base64`) |
| `guidance_scale` | float | — | 2.5 | |
| `num_inference_steps` | int | — | pipeline default | |
| `seed` | int | — | — | reproducibility |

**Output:** `{ "image": "data:image/png;base64,..." }` or `{ "error": "..." }`

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <API_KEY>' \
  -d '{"input":{"prompt":"add sunglasses to the cat","image":"data:image/png;base64,...","guidance_scale":2.5}}'
```

## Deploy (RunPod)

FLUX.1-Kontext-dev is **gated** and large (~24 GB, 12B params). Before deploying:

1. **Accept the license** on HuggingFace: <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev> (agree to the FLUX.1-dev Non-Commercial License).
2. **Create an HF token** (Read) on the account that accepted the license.
3. **Create a RunPod Serverless endpoint** from this GitHub repo (Dockerfile build):
   - **GPU:** 24 GB minimum (set env `CPU_OFFLOAD=1`), 40–48 GB recommended (leave `CPU_OFFLOAD=0` for speed).
   - **Network volume (strongly recommended):** attach one (~30 GB). The worker caches the model under `HF_HOME=/runpod-volume/huggingface`, so it downloads the ~24 GB **once** instead of on every cold start. Without a volume, each cold start re-downloads the model.
   - **Env vars:** `HF_TOKEN=<your token>` (required). Optional: `CPU_OFFLOAD=1`, `MODEL_ID` (override).
4. First request cold-starts a worker and downloads the model to the volume (slow, one time). Subsequent requests are fast.

## Notes
- Model download uses `hf_transfer` for speed. Requires `HF_TOKEN` with license access.
- Output is base64 PNG. Configure S3 offload separately if you prefer URLs.

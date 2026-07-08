# flux2-klein-worker

RunPod serverless worker for **FLUX.2-klein-4B** — image editing (input image + prompt →
edited image) and text-to-image, via diffusers `Flux2KleinPipeline`.

**Why this model:** Apache-2.0 (**free commercial use, no license needed**), **not gated
(no HF token required)**, ~13 GB VRAM (runs on a 16 GB GPU), fast 4-step distilled model,
supports multi-reference editing.

## API

**Input** (`event["input"]`):

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | edit instruction or generation prompt |
| `image` / `image_url` / `image_base64` | string | — | — | one input/reference image (edit mode) |
| `images` | array | — | — | multiple reference images (multi-ref) |
| `guidance_scale` | float | — | 2.5 | |
| `num_inference_steps` | int | — | 4 | klein is distilled (few steps) |
| `height`, `width` | int | — | pipeline default | |
| `seed` | int | — | — | reproducibility |

If no image is provided it runs text→image; with an image it edits/uses it as reference.

**Output:** `{ "image": "data:image/png;base64,..." }` or `{ "error": "..." }`

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <API_KEY>' \
  -d '{"input":{"prompt":"make the background pure white","image":"data:image/png;base64,...","guidance_scale":2.5}}'
```

## Deploy (RunPod)

1. Create a Serverless endpoint from this GitHub repo (Dockerfile build).
2. **GPU:** 16 GB+ (the worker uses `enable_model_cpu_offload` by default via `CPU_OFFLOAD=1`;
   set `CPU_OFFLOAD=0` on a 24 GB+ GPU for a bit more speed).
3. **No network volume needed.** The model (~13 GB) is baked into the image at build time
   (`snapshot_download` into `HF_HOME=/opt/hf`), so there is no runtime download and no
   cold-start re-download. The trade-off is a large image (~15 GB) and a slower first build/pull.
4. **No HF token needed** (Apache-2.0, non-gated). `HF_TOKEN` env is optional.

## Notes
- Requires **diffusers from git** (Flux2KleinPipeline is new) — pinned in `requirements.txt`.
- Output is base64 PNG.

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

**Image source:** URLs are fetched by the worker with a browser `User-Agent`/`Referer`
(alicdn and similar block the default `urllib`/`requests` UA with `420`).

**Output:** `{ "image_url": "https://<r2-public>/klein/<sha256>.png" }` when R2 is configured
(the worker uploads the result to Cloudflare R2 and returns a public URL, content-addressed
by `sha256(source+prompt+params)` with a HEAD cache check). If R2 is **not** configured it
falls back to `{ "image": "data:image/png;base64,..." }`. On error: `{ "error": "..." }`.

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <API_KEY>' \
  -d '{"input":{"prompt":"make the background pure white","image":"data:image/png;base64,...","guidance_scale":2.5}}'
```

## Deploy (RunPod)

1. Create a Serverless endpoint from this GitHub repo (Dockerfile build).
2. **GPU:** 16 GB+ (the worker uses `enable_model_cpu_offload` by default via `CPU_OFFLOAD=1`;
   set `CPU_OFFLOAD=0` on a 24 GB+ GPU for a bit more speed).
3. **Use RunPod Model Caching (not baking).** The model is **not** in the image — the image is
   small (fast to provision). In the endpoint's **Endpoint Configuration**, set the **Model**
   field to `black-forest-labs/FLUX.2-klein-4B`. RunPod pre-positions the model on host machines
   at `/runpod-volume/huggingface-cache/` (matched by `HF_HOME`), so workers cold-start in
   seconds, and you aren't billed while the model downloads. No manual network volume needed.
   (Baking a 13 GB layer made cold starts 2–3 min because that unique layer is pulled fresh per
   worker; RunPod explicitly recommends caching over baking.)
4. **No HF token needed** (Apache-2.0, non-gated). `HF_TOKEN` env is optional.
5. **R2 output (recommended):** set these env vars on the endpoint so results are stored in
   Cloudflare R2 and returned as public URLs (same values as the bg/qwen workers):
   `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_PUBLIC_BASE`,
   and optionally `R2_PREFIX` (default `klein`). Without them the worker returns base64.

## Notes
- Requires **diffusers from git** (Flux2KleinPipeline is new) — pinned in `requirements.txt`.
- Output is base64 PNG.

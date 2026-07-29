# Official FE2E RPX overlay

This image layers the official `AMAP-ML/FE2E` inference release over the
current RPX paper image. Model weights are downloaded at runtime into the
mounted Hugging Face cache; they are not baked into the image.

The runtime retains `gcc` because Triton compiles its CUDA driver helper on
the first actual GPU inference. A successful Python import alone does not
exercise this compilation path.

Pinned provenance:

- source: `AMAP-ML/FE2E@262d4c9d4c37752c984304f57ca7d7066f34ad5d`
- checkpoint: `exander/FE2E@4c52e5e3d133e4111f6099b57879d517dd8677b8`
- `LDRN.safetensors`: `9c2112d3697d6057a60d59b3d83fc0d88ea12d73f933666eded0c5dbf5e0f410`
- `pretrain/step1x-edit-i1258.safetensors`: `f378db311e6db3535bbed367160fda7f65661c1d2474cffeaded47288c00a24a`
- `pretrain/vae.safetensors`: `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`

FE2E produces normalized log-depth. RPX uses `ls_log`: one affine fit from
the raw prediction to log ground truth, pooled over each `(scene, phase)`
cell, followed by exponentiation. Pooling follows the RPX paper protocol;
the upstream FE2E benchmark fits each image independently.

# -*- coding: utf-8 -*-
"""deepfloyd_if_free_tier_google_colab.ipynb

Original file is located at
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/deepfloyd_if_free_tier_google_colab.ipynb

# Running IF with 🧨 diffusers on a Free Tier Google Colab

*Image taken from official IF GitHub repo [here](https://github.com/deep-floyd/IF/blob/release/pics/nabla.jpg)*

## Introduction

IF is a pixel-based text-to-image generation model and was [released in late April 2023 by DeepFloyd](https://github.com/deep-floyd/IF). The model architecture is strongly inspired by [Google's closed-sourced Imagen](https://imagen.research.google/).

IF has two distinct advantages compared to existing text-to-image models like Stable Diffusion:
- The model operates directly in "pixel space" (*i.e.,* on uncompressed images) instead of running the denoising process in the latent space such as [Stable Diffusion](http://hf.co/blog/stable_diffusion).
- The model is trained on outputs of [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl), a more powerful text encoder than [CLIP](https://openai.com/research/clip), used by Stable Diffusion as the text encoder.

As a result, IF is better at generating images with high-frequency details (*e.g.,* human faces, and hands) and is the first open-source image generation model that can reliably generate images with text.

The downside of operating in pixel space and using a more powerful text encoder is that IF has a significantly higher amount of parameters. T5, IF's text-to-image UNet, and IF's upscaler UNet have 4.5B, 4.3B, and 1.2B parameters respectively. Compared this to [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)'s text encoder and UNet having just 400M and 900M parameters respectively.

Nevertheless, it is possible to run IF on consumer hardware if one optimizes the model for low-memory usage. We will show you can do this with 🧨 diffusers in this blog post.

## Optimizing IF to run on memory constrained hardware

The free-tier Google Colab is both CPU RAM constrained (13 GB RAM) as well as GPU VRAM constrained (15 GB RAM for T4) which makes running the whole >10B IF model challenging!

Let's map out the size of IF's model components in full float32 precision:
- [T5-XXL Text Encoder](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/tree/main/text_encoder): 20GB
- [Stage 1 UNet](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/tree/main/unet): 17.2 GB
- [Stage 2 Super Resolution UNet](https://huggingface.co/DeepFloyd/IF-II-L-v1.0/blob/main/pytorch_model.bin): 2.5 GB
- [Stage 3 Super Resolution Model](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler): 3.4 GB

There is no way we can run the model in float32 as the T5 and Stage 1 UNet weights are each larger than the available CPU RAM.

In float16, the component sizes are 11GB, 8.6GB and 1.25GB for T5, Stage1 and Stage2 UNets respectively which is doable for the GPU, but we're still running into CPU memory overflow errors when loading the T5 (some CPU is occupied by other processes).

Therefore, we lower the precision of T5 even more by using `bitsandbytes` 8bit quantization which allows to save the T5 checkpoint with as little as [8 GB](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/blob/main/text_encoder/model.8bit.safetensors).

Now that each components fits individually into both CPU and GPU memory, we need to make sure that components have all the CPU and GPU memory for themselves when needed.

"""
## Install dependencies
"""

! pip install --upgrade \
  diffusers~=0.16 \
  transformers~=4.28 \
  safetensors~=0.3 \
  sentencepiece~=0.1 \
  accelerate~=0.18 \
  bitsandbytes~=0.38 \
  torch~=2.0 -q

## Accepting the license

Before you can use IF, you need to accept its usage conditions. To do so:

- 1. Make sure to have a [Hugging Face account](https://huggingface.co/join) and be loggin in
- 2. Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). Accepting the license on the stage I model card will auto accept for the other IF models.
- 3. Make sure to login locally. Install `huggingface_hub`


!pip install huggingface_hub --upgrade
"""

from huggingface_hub import login

login()

"""## 1. Text-to-image generation

We will walk step by step through text-to-image generation with IF using Diffusers. We will explain briefly APIs and optimizations, but more in-depth explanations can be found in the official documentation for [Diffusers](https://huggingface.co/docs/diffusers/index), [Transformers](https://huggingface.co/docs/transformers/index), [Accelerate](https://huggingface.co/docs/accelerate/index), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

### 1.1 Load text encoder

We will load T5 using 8bit quantization. Transformers directly supports [bitsandbytes](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit) through the `load_in_8bit` flag.

The flag `variant="8bit"` will download pre-quantized weights.

We also use the `device_map` flag to allow `transformers` to offload model layers to the CPU or disk. Transformers big modeling supports arbitrary device maps which can be used to separately load model parameters directly to available devices. Passing `"auto"` will automatically create a device map. See the `transformers` [docs](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map) for more information.
"""

from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    subfolder="text_encoder",
    device_map="auto",
    load_in_8bit=True,
    variant="8bit"
)

"""### 1.2 Create text embeddings

In this case, we pass `None` for the `unet` argument so no UNet will be loaded. This allows us to run the text embedding portion of the diffusion process without loading the UNet into memory.

"""

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    text_encoder=text_encoder, # pass the previously instantiated 8bit text encoder
    unet=None,
    device_map="auto"
)

"""
Let's define a fitting prompt:
"""

prompt = 'a photograph of an astronaut riding a horse holding a sign that says "Pixel\'s in space"'

"""and run it through the 8bit quantized T5 model:"""

prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

"""### 1.3 Free memory

Once the prompt embeddings have been created. We do not need the text encoder anymore. However, it is still in memory on the GPU. We need to remove it so that we can load the UNet.

It's non-trivial to free PyTorch memory. We must garbage collect the Python objects which point to the actual memory allocated on the GPU.

First, use the python keyword `del` to delete all python objects referencing allocated GPU memory
"""

del text_encoder
del pipe

"""The deletion of the python object is not enough to free the GPU memory. Garbage collection is when the actual GPU memory is freed.

Additionally, we will call `torch.cuda.empty_cache()`. This method isn't strictly necessary as the cached cuda memory will be immediately available for further allocations. Emptying the cache allows us to verify in the colab UI that the memory is available.

We'll use a helper function `flush()` to flush memory.
"""

import gc
import torch

def flush():
  gc.collect()
  torch.cuda.empty_cache()

"""and run it"""

flush()

"""### 1.4 Stage 1: The main diffusion process

With our now available GPU memory, we can re-load the `DiffusionPipeline` with only the UNet to run the main diffusion process.

The `variant` and `torch_dtype` flags are used by Diffusers to download and load the weights in 16 bit floating point format.
"""

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
    device_map="auto"
)

"""
Often, we directly pass the text prompt to `DiffusionPipeline.__call__`. However, we previously computed our text embeddings which we can pass instead.

IF also comes with a super resolution diffusion process. Setting `output_type="pt"` will return raw PyTorch tensors instead of a PIL image. This way we can keep the Pytorch tensors on GPU and pass them directly to the stage 2 super resolution pipeline.

Let's define a random generator and run the stage 1 diffusion process.
"""

generator = torch.Generator().manual_seed(1)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    generator=generator,
).images

"""Let's manually convert the raw tensors to PIL and have a sneak peak at the final result. The output of stage 1 is a 64x64 image."""

from diffusers.utils import pt_to_pil

pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]

"""And again, we remove the Python pointer and free CPU and GPU memory:"""

del pipe
flush()

"""### 1.5 Stage 2: Super Resolution 64x64 to 256x256

IF comes with a separate diffusion process for upscaling.

We run each diffusion process with a separate pipeline.

The super resolution pipeline can be loaded with a text encoder if needed. However, we will usually have pre-computed text embeddings from the first IF pipeline. If so, load the pipeline without the text encoder.

Create the pipeline
"""

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None, # no use of text encoder => memory savings!
    variant="fp16",
    torch_dtype=torch.float16,
    device_map="auto"
)

"""and run it, re-using the pre-computed text embeddings"""

image = pipe(
    image=image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    generator=generator,
).images

"""Again we can inspect the intermediate results."""

pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]

"""And again, we delete the Python pointer and free memory"""

del pipe
flush()

"""### 1.6 Stage 3: Super Resolution 256x256 to 1024x1024

The second super resolution model for IF is the previously release [Stability AI's x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).

Let's create the pipeline and load it directly on GPU with `device_map="auto"`.

Note that `device_map="auto"` with certain pipelines will throw errors in diffusers versions from `v0.16-v0.17`. You will either have to upgrade to a later version
if one exists or install from main with `pip install git+https://github.com/huggingface/diffusers.git@main`
"""

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16,
    device_map="auto"
)

"""🧨 diffusers makes independently developed diffusion models easily composable as pipelines can be chained together. Here we can just take the previous PyTorch tensor output and pass it to the tage 3 pipeline as `image=image`.

💡 **Note**: The x4 Upscaler does not use T5 and has [its own text encoder](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main/text_encoder). Therefore, we cannot use the previously created prompt embeddings and instead must pass the original prompt.
"""

pil_image = pipe(prompt, generator=generator, image=image).images

"""Unlike the IF pipelines, the IF watermark will not be added by default to outputs from the Stable Diffusion x4 upscaler pipeline.

We can instead manually apply the watermark. TODO: Removw watermark (borrar la linea ?)
"""

from diffusers.pipelines.deepfloyd_if import IFWatermarker

watermarker = IFWatermarker.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="watermarker")
watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

"""View output image"""

pil_image[0]
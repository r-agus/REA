#El código se ha obtenido de kaggle (se puede ejecutar este mismo código directamente en el entorno de Kaggle): https://www.kaggle.com/code/shonenkov/deepfloyd-if-4-3b-generator-of-pictures
#Primero se debe configurar el entorno y descargar el modelo DeepFloyd IF ejecutando los siguientes comandos
######################################################################################################################################################
#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
#pip install deepfloyd-if==1.0.1rc0
#pip install xformers==0.0.16
#pip install git+https://github.com/openai/CLIP.git --no-deps
#pip list | findstr deepfloyd
#pip list | findstr xformers
#pip list | findstr torch
#nvidia-smi

import os
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
import sys
import random

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII, T5Embedder

#Poner token para confirmar sesión en HuggingFace
hf_token = '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'

device = 'cuda:0'

if_I = IFStageI('IF-I-XL-v1.0', device=device, hf_token=hf_token)

if_II = IFStageII('IF-II-L-v1.0', device='cuda:1', hf_token=hf_token)

if_III = StableStageIII('stable-diffusion-x4-upscaler', device='cuda:1')

t5_embs, prompts = [], []
for prompt_idx in [289, 114, 255, 38]:
    #prompt = open(f'/kaggle/input/t5-prompts-if/{str(prompt_idx).zfill(4)}.txt').read().strip() #Se debe poner el path del dataset de prompts
    #t5_numpy = np.load(f'/kaggle/input/t5-prompts-if/{str(prompt_idx).zfill(4)}.npy')  #Se debe poner el path del dataset de prompts
    prompt = open(f'./t5-prompts-if/{str(prompt_idx).zfill(4)}.txt').read().strip() #Se debe poner el path del dataset de prompts
    t5_numpy = np.load(f'./t5-prompts-if/{str(prompt_idx).zfill(4)}.npy')  #Se debe poner el path del dataset de prompts
    t5_embs.append(torch.from_numpy(t5_numpy).unsqueeze(0))
    prompts.append(prompt)


t5_embs = torch.cat(t5_embs).to(device)
t5_embs.shape


#PRIMERA FASE: Genera una imagen de 64x64 bits
seed = 42

stageI_generations, _meta = if_I.embeddings_to_image(
    t5_embs, seed=seed, batch_repeat=1,
    dynamic_thresholding_p=0.95,
    dynamic_thresholding_c=1.5,
    guidance_scale=7.0,
    sample_loop='ddpm',
    sample_timestep_respacing='smart50',
    image_size=64,
    aspect_ratio="1:1",
    progress=True,
)
pil_images_I = if_I.to_images(stageI_generations)
if_I.show(pil_images_I)

#SEGUNDA FASE: Genera una imagen de 256x256 bits a partir de la anterior
stageII_generations, _meta = if_II.embeddings_to_image(
    stageI_generations,
    t5_embs, seed=seed, batch_repeat=1,
    dynamic_thresholding_p=0.95,
    dynamic_thresholding_c=1.0,
    aug_level=0.25,
    guidance_scale=4.0,
    image_scale=4.0,
    sample_loop='ddpm',
    sample_timestep_respacing='50',
    progress=True,
)
pil_images_II = if_II.to_images(stageII_generations)
if_II.show(pil_images_II)

#TERCERA FASE: Genera una imagen de 1024x1024 bits a partir de la anterior
stageIII_generations = []
for idx in range(len(stageII_generations)):
    if_III_kwargs = {}
    if_III_kwargs['prompt'] = prompts[idx:idx+1]
    if_III_kwargs['low_res'] = stageII_generations[idx:idx+1]
    if_III_kwargs['seed'] = seed
    if_III_kwargs['t5_embs'] = t5_embs[idx:idx+1]
    _stageIII_generations, _meta = if_III.embeddings_to_image(**if_III_kwargs)
    stageIII_generations.append(_stageIII_generations)

stageIII_generations = torch.cat(stageIII_generations, 0)
pil_images_III = if_III.to_images(stageIII_generations)

for pil_img, prompt in zip(pil_images_III, prompts):
    if_I.show([pil_img],size=14)
    print(prompt, '\n'*3)

# Crear un directorio para guardar las imágenes
os.makedirs("generated_images", exist_ok=True)

# Guardar cada imagen como un archivo PNG
for i, pil_img in enumerate(pil_images_III):
    image_path = f"generated_images/image_{i}.png"
    pil_img.save(image_path)

import torch
import json
import os
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL

from nets.transformer_2d import Transformer2DModel
from nets.pipeline import DiTPipeline

# 模型路径
model_path = "model_data/DiT-XL-2-256"

# 初始化DiT的各个组件
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
transformer = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
id2label = json.load(open(os.path.join(model_path, "model_index.json"), "r"))['id2label']

# 初始化DiT的Pipeline
pipe = DiTPipeline(scheduler=scheduler, transformer=transformer, vae=vae, id2label=id2label)
pipe = pipe.to("cuda")

# imagenet种类 对应的 名称
words = ["white shark", "umbrella"]
# 获得imagenet对应的ids
class_ids = pipe.get_label_ids(words)
# 设置seed
generator = torch.manual_seed(42)

# pipeline前传
output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

# 保存图片
for index, image in enumerate(output.images):
    image.save(f"output-{index}.png")
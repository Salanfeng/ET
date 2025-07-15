import torch
from PIL import Image
import numpy as np
# TODO: 需根据 Qwen2.5-VL-3B 官方实现适配

def preprocess_image(image_path, image_size=(224, 224)):
    """图片预处理，返回 tensor"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    # TODO: normalize
    return img.unsqueeze(0)


def get_tokenizer(tokenizer_path):
    # TODO: 适配 Qwen2.5-VL-3B 的 tokenizer
    # 可参考 transformers.AutoTokenizer 或官方实现
    # from transformers import AutoTokenizer
    # return AutoTokenizer.from_pretrained(tokenizer_path)
    pass

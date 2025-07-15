import torch
import os
import json
import logging
from functools import partial
from ...utils import make_output_dir, SimpleADB, make_quantizer
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from .model.static_qwen_vl import QwenVLModel, ModelArgs
from .utils import get_tokenizer, preprocess_image

# 1. 适配 vision encoder 的 example_inputs
# 2. 适配 vision encoder 的量化、导出流程
# 3. 适配多模态输入的推理流程

# TODO: 需根据 Qwen2.5-VL-3B 官方实现完善模型结构和权重加载

def calibrate_vl(example_inputs, user_prompts, image_paths, module, tokenizer, max_seq_len=512, use_i64_token=False):
    """
    多模态校准流程，支持 image + text
    """
    # 假设 example_inputs = (image_tensor, input_ids, attention_mask)
    # image_paths: list[str]，user_prompts: list[str]
    if not isinstance(user_prompts, list):
        user_prompts = [user_prompts]
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    assert len(user_prompts) == len(image_paths), "每个图片应有对应的prompt"

    for prompt, img_path in zip(user_prompts, image_paths):
        # 1. 预处理图片
        image_tensor = preprocess_image(img_path)
        # 2. 文本tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)
        # 3. 校准推理
        with torch.no_grad():
            _ = module(image_tensor, input_ids, attention_mask)
    print("[calibrate_vl] 多模态校准完成")

class SingleQwenVL:
    def __init__(self, qwen_vl_model, pte_filename):
        self.qwen_vl_model = qwen_vl_model
        self.pte_filename = pte_filename
        self.inputs = self.qwen_vl_model.get_example_inputs()
        self.qwen_graph_module = qwen_vl_model
        self.quant_attrs = None
        # ... 其它初始化 ...

    def quantize(self, quant_dtype, args, tokenizer, custom_annotations=()):
        # vision encoder 也需量化
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
        )
        quantizer.add_custom_quant_annotations(custom_annotations)
        fx_graph_module = torch.export.export(
            self.qwen_graph_module, self.inputs, strict=True
        ).module()
        fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
        calibrate_vl(self.inputs, args.prompt[0], args.image_paths, fx_graph_module, tokenizer)
        self.qwen_graph_module = convert_pt2e(fx_graph_module)

    def lowering_modules(self, work_space, use_fp16=False, soc_model=None, num_sharding=1, shared_buffer=False, verbose=False):
        # TODO: 参考 llama.py，适配多模态导出
        pass

    def get_example_inputs(self):
        return self.qwen_vl_model.get_example_inputs()

    def get_quant_attrs(self):
        return self.quant_attrs

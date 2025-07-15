import torch

# TODO: 需根据 Qwen2.5-VL-3B 官方实现完善模型结构
# 下面为结构示意，需用户补充完整

class ModelArgs:
    def __init__(self, **kwargs):
        # vision encoder/LLM参数
        for k, v in kwargs.items():
            setattr(self, k, v)

class QwenVLModel(torch.nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        # TODO: 初始化 vision encoder
        # self.vision_encoder = ...
        # TODO: 初始化 LLM
        # self.llm = ...
        pass

    def forward(self, image_tensor, text_tokens, *args, **kwargs):
        # TODO: 1. 图像预处理与编码
        # image_embeds = self.vision_encoder(image_tensor)
        # 2. 文本编码
        # 3. 融合 image_embeds 和 text_tokens
        # 4. LLM 推理
        # return self.llm(...)
        pass

    def get_example_inputs(self):
        # TODO: 返回 (image_tensor, text_tokens, ...)
        # image_tensor = torch.zeros([1, 3, 224, 224])  # 示例
        # text_tokens = torch.zeros([1, 32], dtype=torch.int32)
        # return (image_tensor, text_tokens)
        pass

    def get_metadata(self):
        # TODO: 返回模型元信息
        return {}

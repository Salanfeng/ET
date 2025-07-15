# Qwen2.5-VL-3B Executorch 导出脚本


## 目录结构

- `qwen_vl.py`：主导出脚本，参考 llama.py，已加注释说明 VLM 适配点。
- `model/static_qwen_vl.py`：Qwen2.5-VL-3B 模型结构定义（含 vision encoder 和 LLM）。
- `utils.py`：图片预处理、tokenizer 封装等工具函数。

## 主要适配点说明

1. **模型结构**
   - 需实现 vision encoder（如 ViT/CLIP）和 LLM 的拼接结构。
   - 需支持加载 vision encoder 和 LLM 的权重。

2. **输入输出**
   - 输入需包含 image tensor（如 [B, C, H, W]）和文本 token。
   - `get_example_inputs` 需返回 (image_tensor, text_tokens, ...)。

3. **量化/导出**
   - vision encoder 也需走量化、导出流程。
   - 校准流程需支持多模态输入。

4. **推理脚本**
   - 支持图片和文本输入，输出格式适配。

## 需要用户自定义/注意的部分

- vision encoder 的结构和权重加载方式（请参考官方 Qwen2.5-VL-3B 实现）。
- 图片预处理流程（如 resize、normalize）。
- tokenizer 选择与适配。
- example_inputs 的构造方式。

---

如需进一步适配其它 VLM，请参考本目录结构和注释。

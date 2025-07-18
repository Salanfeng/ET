# Qwen2.5-VL Export Script

This directory contains scripts for exporting Qwen2.5-VL (Vision-Language) models to run on Qualcomm's QNN backend.

## Overview

The implementation follows the pattern from existing models:
- **Model loading**: Based on Qwen3's HuggingFace loading approach
- **Core implementation**: Based on Llama's updated QNN export pipeline 
- **Vision components**: Custom implementation for handling multimodal inputs

## Features

- Support for Qwen2.5-VL multimodal models
- HuggingFace model loading from pretrained checkpoints
- Quantization support (8a8w, 16a4w, 16a4w_block)
- KV cache and hybrid mode support
- QNN backend lowering for deployment on Qualcomm devices

## Usage

### Basic Export

```bash
python qwen2_5_vl.py \
    --model_dir /path/to/qwen2.5-vl-model \
    --artifact /path/to/output \
    --max_seq_len 128 \
    --model_mode kv \
    --tokenizer_model /path/to/tokenizer.model
```

### With Quantization

```bash
python qwen2_5_vl.py \
    --model_dir /path/to/qwen2.5-vl-model \
    --artifact /path/to/output \
    --max_seq_len 128 \
    --model_mode kv \
    --ptq 16a4w \
    --tokenizer_model /path/to/tokenizer.model
```

### Hybrid Mode (Prefill + Decode)

```bash
python qwen2_5_vl.py \
    --model_dir /path/to/qwen2.5-vl-model \
    --artifact /path/to/output \
    --max_seq_len 512 \
    --prefill_ar_len 128 \
    --model_mode hybrid \
    --tokenizer_model /path/to/tokenizer.model
```

## Arguments

### Required Arguments
- `--model_dir`: Path to HuggingFace Qwen2.5-VL model directory
- `--artifact`: Output directory for exported model files
- `--tokenizer_model`: Path to tokenizer model file

### Model Configuration
- `--max_seq_len`: Maximum sequence length (default: 128)
- `--model_mode`: Model mode - "kv" or "hybrid" (default: kv)
- `--prefill_ar_len`: Prefill autoregressive length for hybrid mode

### Quantization Options
- `--ptq`: Post-training quantization - "8a8w", "16a4w", or "16a4w_block"
- `--embedding_quantize`: Enable embedding quantization
- `--quant_mapping`: Custom quantization mapping
- `--no_groupwise_8w`: Disable groupwise quantization for 8-bit weights
- `--use_16a8w_mixed_activation`: Use mixed activation quantization

### Backend Options
- `--soc_model`: Target SoC model (default: SM8650)
- `--num_sharding`: Number of sharding contexts
- `--shared_buffer`: Enable shared buffer optimization
- `--use_fp16`: Use FP16 precision
- `--verbose`: Enable verbose output

## Model Architecture

The Qwen2.5-VL model consists of:

1. **Vision Encoder**: Vision Transformer for processing images
2. **Language Model**: Qwen2-based decoder for text generation  
3. **Vision-Language Fusion**: Cross-modal attention mechanisms
4. **Multimodal Generation**: Combined text and vision understanding

## File Structure

```
qwen2_5_vl/
├── qwen2_5_vl.py              # Main export script
├── model/
│   ├── __init__.py
│   ├── configuration_qwen2_5_vl.py  # Model configuration
│   └── static_qwen2_5_vl.py         # Model implementation
└── README.md                   # This file
```

## Implementation Details

### Model Loading
- Uses HuggingFace `Qwen2VLForConditionalGeneration.from_pretrained()`
- Converts HF config to custom config format
- Loads pretrained weights with proper key mapping

### Vision Processing
- Patch-based image encoding with Vision Transformer
- Learnable positional embeddings
- Multi-head self-attention layers
- Projection to language model hidden size

### Multimodal Integration
- Vision tokens prepended to text tokens
- Adjusted attention masks for vision-text sequences
- Position encoding adapted for multimodal inputs

### Quantization
- Supports per-channel and block-wise quantization
- Custom annotations for linear layers and matrix operations
- Calibration with multimodal inputs (text + images)

### Export Pipeline
- PyTorch 2.0 export with strict mode
- QNN backend lowering with HTP compiler
- Memory planning for efficient inference
- PTE file generation for deployment

## Dependencies

- PyTorch 2.0+
- transformers
- executorch
- torchao (for quantization)
- pytorch_tokenizers

## Notes

- This implementation is optimized for Qualcomm QNN backend
- Vision components handle standard image inputs (224x224)
- Text-only mode supported by passing None for images
- KV caching implemented for efficient autoregressive generation
- Supports both single-turn and multi-turn conversations

## Troubleshooting

1. **Memory Issues**: Reduce `max_seq_len` or enable `shared_buffer`
2. **Quantization Errors**: Try different quantization schemes or disable groupwise quantization
3. **Vision Processing**: Ensure images are properly preprocessed (normalized, resized)
4. **Model Loading**: Verify HuggingFace model path and tokenizer compatibility

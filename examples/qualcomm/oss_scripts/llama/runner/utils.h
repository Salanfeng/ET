/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstddef>
#include <memory>

// Template struct to hold tensor data and tensor
template <typename T>
struct TensorStruct {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  T* data;
  // data size in bytes
  size_t size;
};


struct PerplexityCalculator {
    double total_log_prob = 0.0;
    size_t total_tokens = 0;
    float current_ppl = 0.0f;

    explicit PerplexityCalculator() = default;

    void reset() {
        total_log_prob = 0.0;
        total_tokens = 0;
        current_ppl = 0.0f;
    }

    void update(double log_prob) {
        total_log_prob += log_prob;
        total_tokens++;
    }

    void finalize() {
        if (total_tokens > 0) {
            current_ppl = static_cast<float>(std::exp(-total_log_prob / total_tokens));
        } else {
            current_ppl = 0.0f;
        }
    }
};
#include <torch/extension.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#if !defined(MUSA_ROUTED_GROUP_ASM)
#error "tn_gm6.cpp requires MUSA_ROUTED_GROUP_ASM"
#endif

#include "mudnn/internal/xmma.h"
#include "mudnn/utils/stl.h"
#include "xmma/matmul/asm_gemm.muh"

namespace {

void check_inputs(const torch::Tensor &grad_output, const torch::Tensor &inp,
                  const torch::Tensor &group_sizes,
                  const torch::Tensor &output) {
  TORCH_CHECK(grad_output.is_privateuseone(), "grad_output must be on MUSA");
  TORCH_CHECK(inp.is_privateuseone(), "inp must be on MUSA");
  TORCH_CHECK(output.is_privateuseone(), "output must be on MUSA");
  TORCH_CHECK(grad_output.device() == inp.device() &&
                  grad_output.device() == output.device(),
              "grad_output, inp, and output must be on the same device");
  TORCH_CHECK(grad_output.scalar_type() == torch::kBFloat16 &&
                  inp.scalar_type() == torch::kBFloat16,
              "GM6 TN inputs must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "GM6 TN output must be float32");
  TORCH_CHECK(grad_output.dim() == 2 && inp.dim() == 2 &&
                  grad_output.size(0) == inp.size(0),
              "GM6 TN inputs must be 2D with equal row counts");
  TORCH_CHECK(output.dim() == 3 &&
                  output.size(1) == grad_output.size(1) &&
                  output.size(2) == inp.size(1),
              "GM6 TN output must have shape [groups, grad_width, input_width]");
  TORCH_CHECK(grad_output.is_contiguous() && inp.is_contiguous() &&
                  output.is_contiguous(),
              "GM6 TN inputs and output must be contiguous");
  TORCH_CHECK(group_sizes.device().is_cpu() &&
                  group_sizes.scalar_type() == torch::kInt64 &&
                  group_sizes.dim() == 1 && group_sizes.is_contiguous(),
              "group_sizes must be a contiguous CPU int64 vector");
  TORCH_CHECK(group_sizes.numel() == output.size(0),
              "group_sizes length must equal output group count");
  TORCH_CHECK(group_sizes.numel() > 1,
              "GM6 variable-K path requires more than one group");
}

}  // namespace

torch::Tensor grouped_wgrad_bf16_fp32(
    const torch::Tensor &grad_output, const torch::Tensor &inp,
    const torch::Tensor &group_sizes, const torch::Tensor &output,
    bool accumulate) {
  check_inputs(grad_output, inp, group_sizes, output);
  c10::musa::MUSAGuard device_guard(grad_output.device());

  const int64_t groups = group_sizes.numel();
  const int64_t m = grad_output.size(1);
  const int64_t n = inp.size(1);
  const auto *sizes = group_sizes.data_ptr<int64_t>();

  thread_local std::vector<::musa::dnn::MatMulParam> params;
  params.resize(groups);
  int64_t total_k = 0;
  for (int64_t i = 0; i < groups; ++i) {
    TORCH_CHECK(sizes[i] > 0, "GM6 TN requires every expert group to be non-empty");
    total_k += sizes[i];
    auto &p = params[i];
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(sizes[i]);
    p.lda = p.m;
    p.ldb = p.n;
    p.ldc = p.n;
    p.ldd = p.n;
    p.stride_a = p.k * p.m;
    p.stride_b = p.k * p.n;
    p.stride_c = p.m * p.n;
    p.stride_d = p.m * p.n;
    p.alpha = 1.0;
    p.beta = accumulate ? 1.0 : 0.0;
    p.gamma = 0.0;
    p.rcp_scale_d = 1.0;
    p.bscale_type = ::musa::dnn::TensorImpl::Type::FLOAT;
    p.ascale_type = ::musa::dnn::TensorImpl::Type::FLOAT;
  }
  TORCH_CHECK(total_k == grad_output.size(0),
              "group_sizes must cover all packed input rows");

  auto &handle = at::GetMudnnHandle();
  auto &handle_impl =
      ::musa::dnn::CastRef<::musa::dnn::HandleImpl>(handle.GetImpl());
  const auto input_dtype = ::musa::dnn::TensorImpl::Type::BFLOAT16;
  const auto output_dtype = ::musa::dnn::TensorImpl::Type::FLOAT;
  ::musa::dnn::MatDesc output_desc{
      output_dtype, output.numel() * output.element_size(),
      static_cast<int>(n), m * n};
  ::musa::dnn::MatDesc grad_desc{
      input_dtype, grad_output.numel() * grad_output.element_size(),
      static_cast<int>(m), static_cast<int64_t>(grad_output.numel())};
  ::musa::dnn::MatDesc inp_desc{
      input_dtype, inp.numel() * inp.element_size(), static_cast<int>(n),
      static_cast<int64_t>(inp.numel())};

  const float beta = accumulate ? 1.0f : 0.0f;
  thread_local ::musa::dnn::GroupGemmParamImpl group_param;
  auto status = ::musa::dnn::AsmKernelTCEGroupGemm(
      handle_impl, params, output.data_ptr(), output_desc,
      grad_output.data_ptr(), grad_desc, inp.data_ptr(), inp_desc,
      output.data_ptr(), output_desc, nullptr, output_desc,
      static_cast<int>(n), static_cast<int>(grad_output.size(0)), true, false,
      1.0f, beta, 0.0f, static_cast<int>(groups), true,
      at::musa::InternalMemAlloc, group_param, -6);
  if (!static_cast<bool>(status)) {
    std::ostringstream details;
    details << "status=" << static_cast<int>(status.status);
    if (status.stack_info) {
      for (const auto &item : *status.stack_info) {
        if (!item.empty()) details << "; " << item;
      }
    }
    TORCH_CHECK(false, "BF16 variable-K TN GM6 wgrad failed: ",
                details.str());
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("grouped_wgrad_bf16_fp32", &grouped_wgrad_bf16_fp32);
}

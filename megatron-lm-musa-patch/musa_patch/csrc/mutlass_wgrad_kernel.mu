#include <ATen/musa/MUSAContext.h>
#include <c10/musa/MUSAException.h>
#include <c10/musa/MUSAGuard.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

// MUTLASS 0.3-dev headers have an order dependency; keep this block intact.
// clang-format off
#include "mutlass/mutlass.h"
#include "mute/tensor.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mutlass/numeric_types.h"
#include "mutlass/epilogue/thread/linear_combination.h"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/kernel/gemm_universal.hpp"
#include "mutlass/util/packed_stride.hpp"
// clang-format on

namespace {

using namespace mute;

using ElementA = mutlass::bfloat16_t;
using ElementB = mutlass::bfloat16_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = mutlass::layout::ColumnMajor;
using LayoutB = mutlass::layout::RowMajor;
using LayoutC = mutlass::layout::RowMajor;
using LayoutD = mutlass::layout::RowMajor;

constexpr int AlignmentA = 2;
constexpr int AlignmentB = 2;
constexpr int AlignmentC = 2;
constexpr int AlignmentD = 2;

using ArchTag = mutlass::arch::Mp31;
using OperatorClass = mutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule = mutlass::gemm::KernelTme;
using EpilogueSchedule = mutlass::epilogue::WithTme;
using ThreadEpilogueOp =
    mutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute,
                                                 ElementC, ElementCompute>;

template <int TileM, int TileN, int TileK, int Stages> struct WgradConfig {
  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using StageCountType = mutlass::gemm::collective::StageCount<Stages>;

  using CollectiveMainloop =
      typename mutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          StageCountType, KernelSchedule>::CollectiveOp;

  using CollectiveEpilogue =
      typename mutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          mutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD,
          AlignmentD, EpilogueSchedule, ThreadEpilogueOp,
          CollectiveMainloop>::CollectiveOp;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// gate_up: dY.T[1536, M] @ X[M, 2048]
using GateUpConfig = WgradConfig<256, 256, 32, 5>;
// down: dY.T[2048, M] @ X[M, 768]
using DownConfig = WgradConfig<256, 128, 32, 4>;
// Current single-node DeepSeek-v3 expert fc1:
// dY.T[4096, M] @ X[M, 7168]. This large-M tuning is deliberately separate
// from the production-like E128 small-M kernels above.
using E32GateUpConfig = WgradConfig<256, 384, 32, 4>;
// Current single-node DeepSeek-v3 expert fc2:
// dY.T[7168, M] @ X[M, 2048]. Four ragged experts share each launch to fill
// the short output-tile grid without changing the per-expert reduction order.
using E32DownConfig = WgradConfig<384, 256, 32, 4>;

constexpr int kMaxE32Experts = 32;
constexpr int kE32GroupsPerLaunch = 4;

template <class Config> struct KGroupedKernelParams {
  typename Config::GemmKernel::Params groups[kE32GroupsPerLaunch];
};
static_assert(sizeof(KGroupedKernelParams<E32GateUpConfig>) <= 4096,
              "MP31 FC1 K-grouped kernel arguments must stay within 4 KiB");
static_assert(sizeof(KGroupedKernelParams<E32DownConfig>) <= 4096,
              "MP31 K-grouped kernel arguments must stay within 4 KiB");

using E32GateUpGemmKernel = typename E32GateUpConfig::GemmKernel;
using E32DownGemmKernel = typename E32DownConfig::GemmKernel;

// Keep the already tuned per-tile MP31 implementation, but expose four expert
// grids to the device scheduler in each launch. GemmUniversal's static tile
// scheduler consumes blockIdx.x and ignores blockIdx.y, so y can select the
// ragged-K expert while x retains the original per-expert tile mapping. Larger
// packs exceed the current MP31 backend's register/argument allocation limit.
MUTLASS_GLOBAL
#ifdef __MUSACC__
__launch_bounds__(E32GateUpGemmKernel::MaxThreadsPerBlock,
                  E32GateUpGemmKernel::MinBlocksPerMultiprocessor)
#endif
    void e32_gate_up_k_grouped_device_kernel(
        MUTLASS_GRID_CONSTANT KGroupedKernelParams<E32GateUpConfig> const
            params) {
  extern __shared__ char
      __attribute__((aligned(E32GateUpGemmKernel::SmemAlignmentBytes))) smem[];
  E32GateUpGemmKernel op;
#define RUN_GROUP(group)                                                       \
  case group:                                                                  \
    op(params.groups[group], smem);                                            \
    break
  switch (blockIdx.y) {
    RUN_GROUP(0);
    RUN_GROUP(1);
    RUN_GROUP(2);
    RUN_GROUP(3);
  default:
    break;
  }
#undef RUN_GROUP
}

MUTLASS_GLOBAL
#ifdef __MUSACC__
__launch_bounds__(E32DownGemmKernel::MaxThreadsPerBlock,
                  E32DownGemmKernel::MinBlocksPerMultiprocessor)
#endif
    void e32_down_k_grouped_device_kernel(
        MUTLASS_GRID_CONSTANT KGroupedKernelParams<E32DownConfig> const
            params) {
  extern __shared__ char
      __attribute__((aligned(E32DownGemmKernel::SmemAlignmentBytes))) smem[];
  E32DownGemmKernel op;
#define RUN_GROUP(group)                                                       \
  case group:                                                                  \
    op(params.groups[group], smem);                                            \
    break
  switch (blockIdx.y) {
    RUN_GROUP(0);
    RUN_GROUP(1);
    RUN_GROUP(2);
    RUN_GROUP(3);
  default:
    break;
  }
#undef RUN_GROUP
}

template <class Config>
void launch_grouped_wgrad(torch::Tensor input, torch::Tensor grad_output,
                          const std::vector<int64_t> &counts,
                          const std::vector<torch::Tensor> &outputs,
                          musaStream_t stream) {
  using GemmKernel = typename Config::GemmKernel;
  using Gemm = typename Config::Gemm;

  // The attribute is invariant for this kernel and device. Configure it once
  // without assuming the process can only ever touch one MUSA device.
  static std::array<std::once_flag, 64> configure_dynamic_smem;
  int const device_index = input.get_device();
  TORCH_CHECK(device_index >= 0 &&
                  device_index <
                      static_cast<int>(configure_dynamic_smem.size()),
              "unsupported MUSA device index: ", device_index);
  std::call_once(configure_dynamic_smem[device_index], [] {
    if constexpr (GemmKernel::SharedStorageSize >= (48 << 10)) {
      C10_MUSA_CHECK(
          musaFuncSetAttribute(mutlass::device_kernel<GemmKernel>,
                               musaFuncAttributeMaxDynamicSharedMemorySize,
                               GemmKernel::SharedStorageSize));
    }
  });

  auto const *x =
      reinterpret_cast<ElementB const *>(input.data_ptr<at::BFloat16>());
  auto const *dy =
      reinterpret_cast<ElementA const *>(grad_output.data_ptr<at::BFloat16>());
  int const n_features = static_cast<int>(grad_output.size(1));
  int const k_features = static_cast<int>(input.size(1));
  float const alpha = 1.0f;
  float const beta = 1.0f;
  int64_t offset = 0;

  for (size_t expert = 0; expert < counts.size(); ++expert) {
    int const tokens = static_cast<int>(counts[expert]);
    if (tokens == 0) {
      continue;
    }

    auto *expert_out = outputs[expert].data_ptr<float>();
    auto problem_shape = make_shape(n_features, k_features, tokens);
    auto stride_a = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideA{}, make_shape(n_features, tokens, 1));
    auto stride_b = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideB{}, make_shape(k_features, tokens, 1));
    auto stride_c = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideC{}, make_shape(n_features, k_features, 1));
    auto stride_d = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideD{}, make_shape(n_features, k_features, 1));

    typename Gemm::Arguments arguments{
        mutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {dy + offset * n_features, stride_a, x + offset * k_features, stride_b},
        {{alpha, beta}, expert_out, stride_c, expert_out, stride_d}};

    auto params = GemmKernel::to_underlying_arguments(arguments, nullptr);
    auto status = Gemm::run(params, stream);
    TORCH_CHECK(status == mutlass::Status::kSuccess,
                "MUTLASS wgrad launch failed for expert ", expert, ": ",
                mutlassGetStatusString(status));
    offset += tokens;
  }
  C10_MUSA_CHECK(musaGetLastError());
}

template <class Config>
void launch_e32_k_grouped_wgrad(torch::Tensor input, torch::Tensor grad_output,
                                const std::vector<int64_t> &counts,
                                const std::vector<torch::Tensor> &outputs,
                                musaStream_t stream) {
  using GemmKernel = typename Config::GemmKernel;
  using Gemm = typename Config::Gemm;
  static_assert(std::is_same_v<Config, E32GateUpConfig> ||
                    std::is_same_v<Config, E32DownConfig>,
                "unsupported MP31 E32 K-grouped config");
  TORCH_CHECK(counts.size() <= kMaxE32Experts,
              "MP31 K-grouped wgrad supports at most ", kMaxE32Experts,
              " experts, got ", counts.size());
  if (std::none_of(counts.begin(), counts.end(),
                   [](int64_t tokens) { return tokens != 0; })) {
    return;
  }

  static std::array<std::once_flag, 64> configure_dynamic_smem;
  int const device_index = input.get_device();
  TORCH_CHECK(device_index >= 0 &&
                  device_index <
                      static_cast<int>(configure_dynamic_smem.size()),
              "unsupported MUSA device index: ", device_index);
  std::call_once(configure_dynamic_smem[device_index], [] {
    if constexpr (GemmKernel::SharedStorageSize >= (48 << 10)) {
      if constexpr (std::is_same_v<Config, E32GateUpConfig>) {
        C10_MUSA_CHECK(
            musaFuncSetAttribute(e32_gate_up_k_grouped_device_kernel,
                                 musaFuncAttributeMaxDynamicSharedMemorySize,
                                 GemmKernel::SharedStorageSize));
      } else {
        C10_MUSA_CHECK(
            musaFuncSetAttribute(e32_down_k_grouped_device_kernel,
                                 musaFuncAttributeMaxDynamicSharedMemorySize,
                                 GemmKernel::SharedStorageSize));
      }
    }
  });

  auto const *x =
      reinterpret_cast<ElementB const *>(input.data_ptr<at::BFloat16>());
  auto const *dy =
      reinterpret_cast<ElementA const *>(grad_output.data_ptr<at::BFloat16>());
  int const n_features = static_cast<int>(grad_output.size(1));
  int const k_features = static_cast<int>(input.size(1));
  float const alpha = 1.0f;
  float const beta = 1.0f;

  std::vector<int64_t> offsets(counts.size());
  int64_t offset = 0;
  for (size_t expert = 0; expert < counts.size(); ++expert) {
    offsets[expert] = offset;
    offset += counts[expert];
  }

  // CUTLASS-style grouped schedulers benefit from presenting the longest
  // reduction-K problems first. Routing changes every microbatch, so sort only
  // the small host metadata and rebuild all tensor descriptors below.
  std::vector<size_t> expert_order(counts.size());
  std::iota(expert_order.begin(), expert_order.end(), size_t{0});
  std::sort(expert_order.begin(), expert_order.end(),
            [&](size_t lhs, size_t rhs) { return counts[lhs] > counts[rhs]; });

  std::array<typename GemmKernel::Params, kMaxE32Experts> active_params;
  int active_groups = 0;
  for (size_t expert : expert_order) {
    int const tokens = static_cast<int>(counts[expert]);
    if (tokens == 0) {
      continue;
    }

    auto *expert_out = outputs[expert].data_ptr<float>();
    auto problem_shape = make_shape(n_features, k_features, tokens);
    auto stride_a = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideA{}, make_shape(n_features, tokens, 1));
    auto stride_b = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideB{}, make_shape(k_features, tokens, 1));
    auto stride_c = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideC{}, make_shape(n_features, k_features, 1));
    auto stride_d = mutlass::make_mute_packed_stride(
        typename GemmKernel::StrideD{}, make_shape(n_features, k_features, 1));

    typename Gemm::Arguments arguments{
        mutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {dy + offsets[expert] * n_features, stride_a,
         x + offsets[expert] * k_features, stride_b},
        {{alpha, beta}, expert_out, stride_c, expert_out, stride_d}};
    active_params[active_groups++] =
        GemmKernel::to_underlying_arguments(arguments, nullptr);
  }

  dim3 grid = GemmKernel::get_grid_shape(active_params[0]);
  TORCH_CHECK(grid.y == 1 && grid.z == 1,
              "unexpected base grid for MP31 K-grouped wgrad: ", grid.x, "x",
              grid.y, "x", grid.z);
  dim3 const block = GemmKernel::get_block_shape();
  for (int first = 0; first < active_groups; first += kE32GroupsPerLaunch) {
    KGroupedKernelParams<Config> grouped_params;
    int const groups_this_launch =
        std::min(kE32GroupsPerLaunch, active_groups - first);
    for (int group = 0; group < groups_this_launch; ++group) {
      grouped_params.groups[group] = active_params[first + group];
    }
    grid.y = static_cast<unsigned int>(groups_this_launch);
    if constexpr (std::is_same_v<Config, E32GateUpConfig>) {
      e32_gate_up_k_grouped_device_kernel<<<
          grid, block, GemmKernel::SharedStorageSize, stream>>>(grouped_params);
    } else {
      e32_down_k_grouped_device_kernel<<<
          grid, block, GemmKernel::SharedStorageSize, stream>>>(grouped_params);
    }
  }
  C10_MUSA_CHECK(musaGetLastError());
}

} // namespace

void mutlass_wgrad_accumulate_musa(torch::Tensor input,
                                   torch::Tensor grad_output,
                                   std::vector<int64_t> counts,
                                   std::vector<torch::Tensor> outputs) {
  TORCH_CHECK(input.is_musa(), "input must be on MUSA");
  TORCH_CHECK(grad_output.is_musa(), "grad_output must be on MUSA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
  TORCH_CHECK(grad_output.scalar_type() == at::kBFloat16,
              "grad_output must be BF16");
  TORCH_CHECK(input.dim() == 2 && grad_output.dim() == 2,
              "input and grad_output must be 2D");
  TORCH_CHECK(input.is_contiguous() && grad_output.is_contiguous(),
              "input and grad_output must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) %
                      (AlignmentB * sizeof(ElementB)) ==
                  0,
              "input pointer does not satisfy MUTLASS alignment");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(grad_output.data_ptr()) %
                      (AlignmentA * sizeof(ElementA)) ==
                  0,
              "grad_output pointer does not satisfy MUTLASS alignment");
  TORCH_CHECK(input.device() == grad_output.device(),
              "input and grad_output must be on the same device");
  TORCH_CHECK(input.size(0) == grad_output.size(0),
              "input and grad_output token counts must match");
  TORCH_CHECK(input.size(1) % AlignmentB == 0 &&
                  grad_output.size(1) % AlignmentA == 0,
              "input feature dimensions do not satisfy MUTLASS alignment");
  TORCH_CHECK(counts.size() == outputs.size(),
              "counts and outputs must have the same expert count");

  int64_t total_tokens = 0;
  for (size_t expert = 0; expert < counts.size(); ++expert) {
    TORCH_CHECK(counts[expert] >= 0, "counts must be non-negative");
    total_tokens += counts[expert];
    auto const &output = outputs[expert];
    TORCH_CHECK(output.is_musa(), "wgrad output must be on MUSA");
    TORCH_CHECK(output.device() == input.device(),
                "wgrad output must be on the input device");
    TORCH_CHECK(output.scalar_type() == at::kFloat,
                "wgrad output must be FP32 main_grad");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == grad_output.size(1) &&
                    output.size(1) == input.size(1),
                "wgrad output has the wrong shape");
    TORCH_CHECK(output.is_contiguous(), "wgrad output must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) %
                        (AlignmentD * sizeof(ElementD)) ==
                    0,
                "wgrad output pointer does not satisfy MUTLASS alignment");
  }
  TORCH_CHECK(total_tokens == input.size(0),
              "sum(counts) must equal the packed input token count");

  c10::musa::MUSAGuard device_guard(input.device());
  musaDeviceProp device_properties{};
  C10_MUSA_CHECK(
      musaGetDeviceProperties(&device_properties, input.get_device()));
  TORCH_CHECK(device_properties.major == 3 && device_properties.minor == 1,
              "MP31 MUTLASS wgrad requires MUSA capability 3.1, got ",
              device_properties.major, ".", device_properties.minor);
  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();
  static bool const use_k_grouped = [] {
    char const *value = std::getenv("MUTLASS_WGRAD_E32_K_GROUPED");
    return value == nullptr || std::string(value) != "0";
  }();
  if (grad_output.size(1) == 4096 && input.size(1) == 7168) {
    if (use_k_grouped) {
      launch_e32_k_grouped_wgrad<E32GateUpConfig>(input, grad_output, counts,
                                                  outputs, stream);
    } else {
      launch_grouped_wgrad<E32GateUpConfig>(input, grad_output, counts, outputs,
                                            stream);
    }
  } else if (grad_output.size(1) == 7168 && input.size(1) == 2048) {
    if (use_k_grouped) {
      launch_e32_k_grouped_wgrad<E32DownConfig>(input, grad_output, counts,
                                                outputs, stream);
    } else {
      launch_grouped_wgrad<E32DownConfig>(input, grad_output, counts, outputs,
                                          stream);
    }
  } else if (input.size(1) <= 1024) {
    launch_grouped_wgrad<DownConfig>(input, grad_output, counts, outputs,
                                     stream);
  } else {
    launch_grouped_wgrad<GateUpConfig>(input, grad_output, counts, outputs,
                                       stream);
  }
}

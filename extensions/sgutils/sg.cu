// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "helper_math.h"
#include "utils.h"

using namespace math;

constexpr float TWOPI = 6.28318530718;
constexpr float INV2PI = 0.15915494309;
constexpr float SQRT2PI23 = 3.03352966508;
constexpr float INVSQRT2PI23 = 0.32964899322;

__device__ __inline__ float square(float v) {
  return v * v;
}

__global__ void evaluate_gaussian_fwd_kernel(
    const float3* __restrict__ lobe_dirs,
    const float* __restrict__ lobe_sigmas,
    const float3* __restrict__ light_values,
    const float3* __restrict__ light_pts,
    const float3* __restrict__ prim_pts,
    const int* __restrict__ n_lights,
    float3* __restrict__ integral,
    const int N,
    const int D,
    const int L,
    const int w_type) {
  const int n = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;

  if (n < N && d < D) {
    const float3 dir = lobe_dirs[n * D + d];
    const float sigma = lobe_sigmas[n * D + d];
    const float3 ppts = prim_pts[n * D + d];

    float3 sum = make_float3(0.f, 0.f, 0.f);
    const int nL = n_lights[n];
    for (int l = 0; l < nL; ++l) {
      const float3 env = light_values[n * L + l];
      const float3 lpts = light_pts[n * L + l];
      float3 ldir = lpts - ppts;
      ldir /= sqrtf(dot(ldir, ldir));
      const float cos_dot = clamp(dot(ldir, dir), -1.f, 1.f);
      float weight = 0.f;
      const float angle = acosf(cos_dot);
      switch (w_type) {
        case 0:
          weight = __expf(-0.5f * square(angle / sigma)) / (sigma * SQRT2PI23);
          break;
        case 1:
          weight = __expf(-0.5f * square(angle / sigma));
          break;
        case 2:
          weight = __expf((cos_dot - 1.f) / sigma) / (sigma * TWOPI);
          break;
        case 3:
          weight = __expf((cos_dot - 1.f) / sigma);
          break;
      }
      sum += env * weight;
    }

    integral[n * D + d] = sum;
  }
}

__global__ void evaluate_gaussian_bwd_kernel(
    const float3* __restrict__ lobe_dirs,
    const float* __restrict__ lobe_sigmas,
    const float3* __restrict__ light_values,
    const float3* __restrict__ light_pts,
    const float3* __restrict__ prim_pts,
    const int* __restrict__ n_lights,
    const float3* __restrict__ grad_integral,
    float3* __restrict__ grad_dirs,
    float* __restrict__ grad_lobe_sigmas,
    float3* __restrict__ grad_light_values,
    const int N,
    const int D,
    const int L,
    const int w_type) {
  const int n = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (n < N && d < D) {
    const float3 dL_integ = grad_integral[n * D + d];
    const float3 dir = lobe_dirs[n * D + d];
    const float sigma = lobe_sigmas[n * D + d];
    const float3 ppts = prim_pts[n * D + d];

    float3 dL_dir = make_float3(0.f, 0.f, 0.f);
    float dL_sigma = 0.f;

    const int nL = n_lights[n];
    for (int l = 0; l < nL; ++l) {
      const float3 env = light_values[n * L + l];
      const float3 lpts = light_pts[n * L + l];
      float3 ldir = lpts - ppts;
      ldir /= sqrtf(dot(ldir, ldir));
      const float cos_dot = dot(ldir, dir);
      const float cos_dot_clamp = clamp(cos_dot, -1.f, 1.f);
      const float angle = acosf(cos_dot_clamp);
      float weight = 0.f;
      float dL_cos_dot = 0.f;
      float dL_angle = 0.f;
      float dL_weight = 0.f;
      float expval = 0.f;
      switch (w_type) {
        case 0:
          expval = __expf(-0.5f * square(angle / sigma));
          weight = expval / (sigma * SQRT2PI23);
          dL_weight = dot(dL_integ, env);
          dL_sigma += dL_weight *
              ((expval * INVSQRT2PI23 * (square(angle) - square(sigma))) /
               (square(sigma) * square(sigma)));

          dL_angle = dL_weight * -((INVSQRT2PI23 * angle * expval) / (square(sigma) * sigma));
          dL_cos_dot = dL_angle *
              ((cos_dot > -1.f && cos_dot < 1.f) ? (-1.f / sqrtf(1.f - square(cos_dot))) : -20.f);
          break;
        case 1:
          expval = __expf(-0.5f * square(angle / sigma));
          weight = expval;
          dL_weight = dot(dL_integ, env);
          dL_sigma += dL_weight * ((expval * square(angle)) / (sigma * square(sigma)));

          dL_angle = dL_weight * -((angle * expval) / square(sigma));
          dL_cos_dot = dL_angle *
              ((cos_dot > -1.f && cos_dot < 1.f) ? (-1.f / sqrtf(1.f - square(cos_dot))) : -20.f);
          break;
        case 2:
          expval = __expf((cos_dot_clamp - 1.f) / sigma);
          weight = expval / (sigma * TWOPI);
          dL_weight = dot(dL_integ, env);
          dL_sigma += dL_weight *
              ((expval * INV2PI * ((1.f - cos_dot_clamp) - sigma)) / (sigma * square(sigma)));

          dL_cos_dot = dL_weight * INV2PI * expval / square(sigma);
          break;
        case 3:
          expval = __expf((cos_dot_clamp - 1.f) / sigma);
          weight = expval;

          dL_weight = dot(dL_integ, env);
          dL_sigma += dL_weight * ((expval * (1.f - cos_dot_clamp) / square(sigma)));

          dL_cos_dot = dL_weight * expval / sigma;
          break;
      }

      dL_dir.x += dL_cos_dot * ldir.x;
      dL_dir.y += dL_cos_dot * ldir.y;
      dL_dir.z += dL_cos_dot * ldir.z;

      if (grad_light_values) {
        atomicAdd((float*)grad_light_values + n * L * 3 + l * 3 + 0, dL_integ.x * weight);
        atomicAdd((float*)grad_light_values + n * L * 3 + l * 3 + 1, dL_integ.y * weight);
        atomicAdd((float*)grad_light_values + n * L * 3 + l * 3 + 2, dL_integ.z * weight);
      }
    }

    grad_lobe_sigmas[n * D + d] = dL_sigma;
    grad_dirs[n * D + d] = dL_dir;
  }
}

std::vector<torch::Tensor> evaluate_gaussian_fwd(
    const torch::Tensor& lobe_dirs,
    const torch::Tensor& lobe_sigmas,
    const torch::Tensor& light_values,
    const torch::Tensor& light_pts,
    const torch::Tensor& prim_pts,
    const torch::Tensor& n_lights,
    torch::Tensor integral,
    const int64_t w_type) {
  CHECK_INPUT(lobe_dirs);
  CHECK_INPUT(lobe_sigmas);
  CHECK_INPUT(light_values);
  CHECK_INPUT(light_pts);
  CHECK_INPUT(prim_pts);
  CHECK_INPUT(n_lights);
  CHECK_INPUT(integral);

  const uint32_t N = lobe_dirs.size(0);
  const uint32_t D = lobe_dirs.size(1);
  const uint32_t L = light_values.size(1);

  AT_ASSERTM(lobe_sigmas.size(0) == N, "Batch dim mismatch for lobe_sigmas.");
  AT_ASSERTM(light_values.size(0) == N, "Batch dim mismatch for light_values.");
  AT_ASSERTM(light_pts.size(0) == N, "Batch dim mismatch for light_pts.");
  AT_ASSERTM(prim_pts.size(0) == N, "Batch dim mismatch for prim_pts.");
  AT_ASSERTM(integral.size(0) == N, "Batch dim mismatch for integral.");

  c10::cuda::CUDAGuard deviceGuard{lobe_dirs.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const uint32_t blocksize = 128;
  const dim3 gridsize = {N, (D + blocksize - 1) / blocksize, 1};

  evaluate_gaussian_fwd_kernel<<<gridsize, blocksize, 0, stream>>>(
      reinterpret_cast<const float3*>(lobe_dirs.data_ptr<float>()),
      lobe_sigmas.data_ptr<float>(),
      reinterpret_cast<const float3*>(light_values.data_ptr<float>()),
      reinterpret_cast<const float3*>(light_pts.data_ptr<float>()),
      reinterpret_cast<const float3*>(prim_pts.data_ptr<float>()),
      reinterpret_cast<const int*>(n_lights.data_ptr<int>()),
      reinterpret_cast<float3*>(integral.data_ptr<float>()),
      N,
      D,
      L,
      w_type);

  return {};
}

std::vector<torch::Tensor> evaluate_gaussian_bwd(
    const torch::Tensor& lobe_dirs,
    const torch::Tensor& lobe_sigmas,
    const torch::Tensor& light_values,
    const torch::Tensor& light_pts,
    const torch::Tensor& prim_pts,
    const torch::Tensor& n_lights,
    const torch::Tensor& grad_integral,
    torch::Tensor& grad_dirs,
    torch::Tensor& grad_lobe_sigmas,
    torch::optional<torch::Tensor> grad_light_values,
    const int64_t w_type) {
  CHECK_INPUT(lobe_dirs);
  CHECK_INPUT(lobe_sigmas);
  CHECK_INPUT(light_values);
  CHECK_INPUT(light_pts);
  CHECK_INPUT(prim_pts);
  CHECK_INPUT(n_lights);
  CHECK_INPUT(grad_integral);
  CHECK_INPUT(grad_dirs);
  CHECK_INPUT(grad_lobe_sigmas);
  if (grad_light_values) {
    CHECK_INPUT(*grad_light_values);
  }

  const uint32_t N = lobe_dirs.size(0);
  const uint32_t D = lobe_dirs.size(1);
  const uint32_t L = light_values.size(1);

  c10::cuda::CUDAGuard deviceGuard{lobe_dirs.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const uint32_t blocksize = 128;
  const dim3 gridsize = {N, (D + blocksize - 1) / blocksize, 1};

  evaluate_gaussian_bwd_kernel<<<gridsize, blocksize, 0, stream>>>(
      reinterpret_cast<const float3*>(lobe_dirs.data_ptr<float>()),
      lobe_sigmas.data_ptr<float>(),
      reinterpret_cast<const float3*>(light_values.data_ptr<float>()),
      reinterpret_cast<const float3*>(light_pts.data_ptr<float>()),
      reinterpret_cast<const float3*>(prim_pts.data_ptr<float>()),
      reinterpret_cast<const int*>(n_lights.data_ptr<int>()),
      reinterpret_cast<const float3*>(grad_integral.data_ptr<float>()),
      reinterpret_cast<float3*>(grad_dirs.data_ptr<float>()),
      grad_lobe_sigmas.data_ptr<float>(),
      grad_light_values ? reinterpret_cast<float3*>(grad_light_values->data_ptr<float>()) : nullptr,
      N,
      D,
      L,
      w_type);

  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluate_gaussian_fwd", &evaluate_gaussian_fwd, "");
  m.def("evaluate_gaussian_bwd", &evaluate_gaussian_bwd, "");
}

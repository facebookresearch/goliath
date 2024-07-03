// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <limits>

// The header provides uniform GLSL-like math API for the following three cases:
//  - non-NVCC compiler
//  - NVCC compiler, host code
//  - NVCC compiler, device code
// Designed to be a more flexible replacement of similar header from NVidia
#ifndef __CUDACC__
#define MH_NON_NVCC
#else
#define MH_NVCC
#ifdef __CUDA_ARCH__
#define MH_NVCC_DEVICE
#else
#define MH_NVCC_HOST
#endif
#endif

#if defined(MH_NVCC_HOST) || defined(MH_NON_NVCC)
#define HOST_DEVICE_DISPATCH(HOST_CODE, DEVICE_CODE) (HOST_CODE)
#elif defined(MH_NVCC_DEVICE)
#define HOST_DEVICE_DISPATCH(HOST_CODE, DEVICE_CODE) (DEVICE_CODE)
#else
#error Dispatch failed
#endif

// if not NVCC, need to include cmath, since certain builtin NVCC functions have
// equivalent ones in cmath
#ifdef MH_NON_NVCC
#include <cmath>
#endif

#define CHD_FUNC constexpr inline __host__ __device__
#define HD_FUNC inline __host__ __device__

namespace math {

template <typename T>
struct epsilon;

template <>
struct epsilon<float> {
  static constexpr float value = 1e-8f;
};

template <>
struct epsilon<double> {
  static constexpr double value = 1e-16;
};

// Host and device version of saturate
// Note that unfortunately `__saturatef` aka `saturate` is a device only
// function. If you do `using namespace math` you would still have to use math
// namespace for scalars: math::saturate
HD_FUNC float saturate(float a) {
  return HOST_DEVICE_DISPATCH(
      fminf(fmaxf(a, 0.0f), 1.0f),
      __saturatef(a) // __saturatef is a device only function
  );
}
// There is no CUDA intrinsic for saturate for double type
HD_FUNC double saturate(double a) {
  return fmin(fmax(a, 0.0), 1.0);
}

// If NVCC then use builtin abs/max/min/sqrt/rsqrt.
// All of them have overloads for ints, floats, and doubles,defined in
// `cuda/crt/math_functions.hpp` thus no need for explicit usage of e.g. fabsf
#if defined(MH_NVCC)
using ::abs;
using ::max;
using ::min;
using ::rsqrt;
using ::sqrt;
#else
// Otherwise use the ones from cmath
using std::abs;
using std::max;
using std::min;
using std::sqrt;

inline double rsqrt(double v) {
  return 1.0 / std::sqrt(v);
}
inline float rsqrt(float v) {
  return 1.0f / std::sqrt(v);
}
#endif

namespace detail {
// Provide overloads of norm3d/norm4d for floats and doubles
HD_FUNC float norm3d(float a, float b, float c) {
  return HOST_DEVICE_DISPATCH(
      sqrt(a * a + b * b + c * c), ::norm3df(a, b, c) // norm3df is device only
  );
}
HD_FUNC double norm3d(double a, double b, double c) {
  return HOST_DEVICE_DISPATCH(
      sqrt(a * a + b * b + c * c), ::norm3d(a, b, c) // norm3d is device only
  );
}
HD_FUNC float rnorm3d(float a, float b, float c) {
  return HOST_DEVICE_DISPATCH(
      1.0f / sqrt(a * a + b * b + c * c), ::rnorm3df(a, b, c) // rnorm3df is device only
  );
}
HD_FUNC double rnorm3d(double a, double b, double c) {
  return HOST_DEVICE_DISPATCH(
      1.0 / sqrt(a * a + b * b + c * c), ::rnorm3d(a, b, c) // rnorm3d is device only
  );
}
HD_FUNC float norm4d(float a, float b, float c, float d) {
  return HOST_DEVICE_DISPATCH(
      sqrt(a * a + b * b + c * c + d * d), ::norm4df(a, b, c, d) // norm4df is device only
  );
}
HD_FUNC double norm4d(double a, double b, double c, double d) {
  return HOST_DEVICE_DISPATCH(
      sqrt(a * a + b * b + c * c + d * d), ::norm4d(a, b, c, d) // norm4d is device only
  );
}
HD_FUNC float rnorm4d(float a, float b, float c, float d) {
  return HOST_DEVICE_DISPATCH(
      1.0f / sqrt(a * a + b * b + c * c + d * d), ::rnorm4df(a, b, c, d) // rnorm4df is device only
  );
}
HD_FUNC double rnorm4d(double a, double b, double c, double d) {
  return HOST_DEVICE_DISPATCH(
      1.0 / sqrt(a * a + b * b + c * c + d * d), ::rnorm4d(a, b, c, d) // rnorm4d is device only
  );
}
} // namespace detail

// Unary operators
#define UNARY_OP(T, T2, T3, T4)        \
  CHD_FUNC T2 operator+(T2 const& v) { \
    return v;                          \
  }                                    \
  CHD_FUNC T2 operator-(T2 const& v) { \
    return {-v.x, -v.y};               \
  }                                    \
  CHD_FUNC T3 operator+(T3 const& v) { \
    return v;                          \
  }                                    \
  CHD_FUNC T3 operator-(T3 const& v) { \
    return {-v.x, -v.y, -v.z};         \
  }                                    \
  CHD_FUNC T4 operator+(T4 const& v) { \
    return v;                          \
  }                                    \
  CHD_FUNC T4 operator-(T4 const& v) { \
    return {-v.x, -v.y, -v.z, -v.w};   \
  }

// -- Binary arithmetic operators --
#define BINARY_ARITHM_OP(T, T2, T3, T4)                              \
  CHD_FUNC T2 operator+(T2 const& v, T scalar) {                     \
    return {v.x + scalar, v.y + scalar};                             \
  }                                                                  \
  CHD_FUNC T2 operator+(T scalar, T2 const& v) {                     \
    return {scalar + v.x, scalar + v.y};                             \
  }                                                                  \
  CHD_FUNC T2 operator+(T2 const& v1, T2 const& v2) {                \
    return {v1.x + v2.x, v1.y + v2.y};                               \
  }                                                                  \
  CHD_FUNC T2 operator+=(T2& v, T scalar) {                          \
    v.x += scalar;                                                   \
    v.y += scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator+=(T2& v, T2 const& v2) {                      \
    v.x += v2.x;                                                     \
    v.y += v2.y;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator-(T2 const& v, T scalar) {                     \
    return {v.x - scalar, v.y - scalar};                             \
  }                                                                  \
  CHD_FUNC T2 operator-(T scalar, T2 const& v) {                     \
    return {scalar - v.x, scalar - v.y};                             \
  }                                                                  \
  CHD_FUNC T2 operator-(T2 const& v1, T2 const& v2) {                \
    return {v1.x - v2.x, v1.y - v2.y};                               \
  }                                                                  \
  CHD_FUNC T2 operator-=(T2& v, T scalar) {                          \
    v.x -= scalar;                                                   \
    v.y -= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator-=(T2& v, T2 const& v2) {                      \
    v.x -= v2.x;                                                     \
    v.y -= v2.y;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator*(T2 const& v, T scalar) {                     \
    return {v.x * scalar, v.y * scalar};                             \
  }                                                                  \
  CHD_FUNC T2 operator*(T scalar, T2 const& v) {                     \
    return {scalar * v.x, scalar * v.y};                             \
  }                                                                  \
  CHD_FUNC T2 operator*(T2 const& v1, T2 const& v2) {                \
    return {v1.x * v2.x, v1.y * v2.y};                               \
  }                                                                  \
  CHD_FUNC T2 operator*=(T2& v, T scalar) {                          \
    v.x *= scalar;                                                   \
    v.y *= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator*=(T2& v, T2 const& v2) {                      \
    v.x *= v2.x;                                                     \
    v.y *= v2.y;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator/(T2 const& v, T scalar) {                     \
    return {v.x / scalar, v.y / scalar};                             \
  }                                                                  \
  CHD_FUNC T2 operator/(T scalar, T2 const& v) {                     \
    return {scalar / v.x, scalar / v.y};                             \
  }                                                                  \
  CHD_FUNC T2 operator/(T2 const& v1, T2 const& v2) {                \
    return {v1.x / v2.x, v1.y / v2.y};                               \
  }                                                                  \
  CHD_FUNC T2 operator/=(T2& v, T scalar) {                          \
    v.x /= scalar;                                                   \
    v.y /= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T2 operator/=(T2& v, T2 const& v2) {                      \
    v.x /= v2.x;                                                     \
    v.y /= v2.y;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator+(T3 const& v, T scalar) {                     \
    return {v.x + scalar, v.y + scalar, v.z + scalar};               \
  }                                                                  \
  CHD_FUNC T3 operator+(T scalar, T3 const& v) {                     \
    return {scalar + v.x, scalar + v.y, scalar + v.z};               \
  }                                                                  \
  CHD_FUNC T3 operator+(T3 const& v1, T3 const& v2) {                \
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};                  \
  }                                                                  \
  CHD_FUNC T3 operator+=(T3& v, T scalar) {                          \
    v.x += scalar;                                                   \
    v.y += scalar;                                                   \
    v.z += scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator+=(T3& v, T3 const& v2) {                      \
    v.x += v2.x;                                                     \
    v.y += v2.y;                                                     \
    v.z += v2.z;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator-(T3 const& v, T scalar) {                     \
    return {v.x - scalar, v.y - scalar, v.z - scalar};               \
  }                                                                  \
  CHD_FUNC T3 operator-(T scalar, T3 const& v) {                     \
    return {scalar - v.x, scalar - v.y, scalar - v.z};               \
  }                                                                  \
  CHD_FUNC T3 operator-(T3 const& v1, T3 const& v2) {                \
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};                  \
  }                                                                  \
  CHD_FUNC T3 operator-=(T3& v, T scalar) {                          \
    v.x -= scalar;                                                   \
    v.y -= scalar;                                                   \
    v.z -= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator-=(T3& v, T3 const& v2) {                      \
    v.x -= v2.x;                                                     \
    v.y -= v2.y;                                                     \
    v.z -= v2.z;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator*(T3 const& v, T scalar) {                     \
    return {v.x * scalar, v.y * scalar, v.z * scalar};               \
  }                                                                  \
  CHD_FUNC T3 operator*(T scalar, T3 const& v) {                     \
    return {scalar * v.x, scalar * v.y, scalar * v.z};               \
  }                                                                  \
  CHD_FUNC T3 operator*(T3 const& v1, T3 const& v2) {                \
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};                  \
  }                                                                  \
  CHD_FUNC T3 operator*=(T3& v, T scalar) {                          \
    v.x *= scalar;                                                   \
    v.y *= scalar;                                                   \
    v.z *= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator*=(T3& v, T3 const& v2) {                      \
    v.x *= v2.x;                                                     \
    v.y *= v2.y;                                                     \
    v.z *= v2.z;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator/(T3 const& v, T scalar) {                     \
    return {v.x / scalar, v.y / scalar, v.z / scalar};               \
  }                                                                  \
  CHD_FUNC T3 operator/(T scalar, T3 const& v) {                     \
    return {scalar / v.x, scalar / v.y, scalar / v.z};               \
  }                                                                  \
  CHD_FUNC T3 operator/(T3 const& v1, T3 const& v2) {                \
    return {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};                  \
  }                                                                  \
  CHD_FUNC T3 operator/=(T3& v, T scalar) {                          \
    v.x /= scalar;                                                   \
    v.y /= scalar;                                                   \
    v.z /= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T3 operator/=(T3& v, T3 const& v2) {                      \
    v.x /= v2.x;                                                     \
    v.y /= v2.y;                                                     \
    v.z /= v2.z;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator+(T4 const& v, T scalar) {                     \
    return {v.x + scalar, v.y + scalar, v.z + scalar, v.w + scalar}; \
  }                                                                  \
  CHD_FUNC T4 operator+(T scalar, T4 const& v) {                     \
    return {scalar + v.x, scalar + v.y, scalar + v.z, scalar + v.w}; \
  }                                                                  \
  CHD_FUNC T4 operator+(T4 const& v1, T4 const& v2) {                \
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w};     \
  }                                                                  \
  CHD_FUNC T4 operator+=(T4& v, T scalar) {                          \
    v.x += scalar;                                                   \
    v.y += scalar;                                                   \
    v.z += scalar;                                                   \
    v.w += scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator+=(T4& v, T4 const& v2) {                      \
    v.x += v2.x;                                                     \
    v.y += v2.y;                                                     \
    v.z += v2.z;                                                     \
    v.w += v2.w;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator-(T4 const& v, T scalar) {                     \
    return {v.x - scalar, v.y - scalar, v.z - scalar, v.w - scalar}; \
  }                                                                  \
  CHD_FUNC T4 operator-(T scalar, T4 const& v) {                     \
    return {scalar - v.x, scalar - v.y, scalar - v.z, scalar - v.w}; \
  }                                                                  \
  CHD_FUNC T4 operator-(T4 const& v1, T4 const& v2) {                \
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w};     \
  }                                                                  \
  CHD_FUNC T4 operator-=(T4& v, T scalar) {                          \
    v.x -= scalar;                                                   \
    v.y -= scalar;                                                   \
    v.z -= scalar;                                                   \
    v.w -= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator-=(T4& v, T4 const& v2) {                      \
    v.x -= v2.x;                                                     \
    v.y -= v2.y;                                                     \
    v.z -= v2.z;                                                     \
    v.w -= v2.w;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator*(T4 const& v, T scalar) {                     \
    return {v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar}; \
  }                                                                  \
  CHD_FUNC T4 operator*(T scalar, T4 const& v) {                     \
    return {scalar * v.x, scalar * v.y, scalar * v.z, scalar * v.w}; \
  }                                                                  \
  CHD_FUNC T4 operator*(T4 const& v1, T4 const& v2) {                \
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w};     \
  }                                                                  \
  CHD_FUNC T4 operator*=(T4& v, T scalar) {                          \
    v.x *= scalar;                                                   \
    v.y *= scalar;                                                   \
    v.z *= scalar;                                                   \
    v.w *= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator*=(T4& v, T4 const& v2) {                      \
    v.x *= v2.x;                                                     \
    v.y *= v2.y;                                                     \
    v.z *= v2.z;                                                     \
    v.w *= v2.w;                                                     \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator/(T4 const& v, T scalar) {                     \
    return {v.x / scalar, v.y / scalar, v.z / scalar, v.w / scalar}; \
  }                                                                  \
  CHD_FUNC T4 operator/(T scalar, T4 const& v) {                     \
    return {scalar / v.x, scalar / v.y, scalar / v.z, scalar / v.w}; \
  }                                                                  \
  CHD_FUNC T4 operator/(T4 const& v1, T4 const& v2) {                \
    return {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w};     \
  }                                                                  \
  CHD_FUNC T4 operator/=(T4& v, T scalar) {                          \
    v.x /= scalar;                                                   \
    v.y /= scalar;                                                   \
    v.z /= scalar;                                                   \
    v.w /= scalar;                                                   \
    return v;                                                        \
  }                                                                  \
  CHD_FUNC T4 operator/=(T4& v, T4 const& v2) {                      \
    v.x /= v2.x;                                                     \
    v.y /= v2.y;                                                     \
    v.z /= v2.z;                                                     \
    v.w /= v2.w;                                                     \
    return v;                                                        \
  }

#define BINARY_INT_OP(T, T2, T3, T4)                                                     \
  CHD_FUNC T2 operator%(T2 const& v, T scalar) {                                         \
    return {(T)(v.x % scalar), (T)(v.y % scalar)};                                       \
  }                                                                                      \
  CHD_FUNC T2 operator%(T scalar, T2 const& v) {                                         \
    return {(T)(scalar % v.x), (T)(scalar % v.y)};                                       \
  }                                                                                      \
  CHD_FUNC T2 operator%(T2 const& v1, T2 const& v2) {                                    \
    return {(T)(v1.x % v2.x), (T)(v1.y % v2.y)};                                         \
  }                                                                                      \
  CHD_FUNC T3 operator%(T3 const& v, T scalar) {                                         \
    return {(T)(v.x % scalar), (T)(v.y % scalar), (T)(v.z % scalar)};                    \
  }                                                                                      \
  CHD_FUNC T3 operator%(T scalar, T3 const& v) {                                         \
    return {(T)(scalar % v.x), (T)(scalar % v.y), (T)(scalar % v.z)};                    \
  }                                                                                      \
  CHD_FUNC T3 operator%(T3 const& v1, T3 const& v2) {                                    \
    return {(T)(v1.x % v2.x), (T)(v1.y % v2.y), (T)(v1.z % v2.z)};                       \
  }                                                                                      \
  CHD_FUNC T4 operator%(T4 const& v, T scalar) {                                         \
    return {(T)(v.x % scalar), (T)(v.y % scalar), (T)(v.z % scalar), (T)(v.w % scalar)}; \
  }                                                                                      \
  CHD_FUNC T4 operator%(T scalar, T4 const& v) {                                         \
    return {(T)(scalar % v.x), (T)(scalar % v.y), (T)(scalar % v.z), (T)(scalar % v.w)}; \
  }                                                                                      \
  CHD_FUNC T4 operator%(T4 const& v1, T4 const& v2) {                                    \
    return {(T)(v1.x % v2.x), (T)(v1.y % v2.y), (T)(v1.z % v2.z), (T)(v1.w % v2.w)};     \
  }

// -- Binary bit operators --
#define BINARY_BIT_OP(T, T2, T3, T4)                                     \
  CHD_FUNC T2 operator&(T2 const& v, T scalar) {                         \
    return {v.x & scalar, v.y & scalar};                                 \
  }                                                                      \
  CHD_FUNC T2 operator&(T scalar, T2 const& v) {                         \
    return {scalar & v.x, scalar & v.y};                                 \
  }                                                                      \
  CHD_FUNC T2 operator&(T2 const& v1, T2 const& v2) {                    \
    return {v1.x & v2.x, v1.y & v2.y};                                   \
  }                                                                      \
  CHD_FUNC T2 operator|(T2 const& v, T scalar) {                         \
    return {v.x | scalar, v.y | scalar};                                 \
  }                                                                      \
  CHD_FUNC T2 operator|(T scalar, T2 const& v) {                         \
    return {scalar | v.x, scalar | v.y};                                 \
  }                                                                      \
  CHD_FUNC T2 operator|(T2 const& v1, T2 const& v2) {                    \
    return {v1.x | v2.x, v1.y | v2.y};                                   \
  }                                                                      \
  CHD_FUNC T2 operator^(T2 const& v, T scalar) {                         \
    return {v.x ^ scalar, v.y ^ scalar};                                 \
  }                                                                      \
  CHD_FUNC T2 operator^(T scalar, T2 const& v) {                         \
    return {scalar ^ v.x, scalar ^ v.y};                                 \
  }                                                                      \
  CHD_FUNC T2 operator^(T2 const& v1, T2 const& v2) {                    \
    return {v1.x ^ v2.x, v1.y ^ v2.y};                                   \
  }                                                                      \
  CHD_FUNC T2 operator<<(T2 const& v, T scalar) {                        \
    return {v.x << scalar, v.y << scalar};                               \
  }                                                                      \
  CHD_FUNC T2 operator<<(T scalar, T2 const& v) {                        \
    return {scalar << v.x, scalar << v.y};                               \
  }                                                                      \
  CHD_FUNC T2 operator<<(T2 const& v1, T2 const& v2) {                   \
    return {v1.x << v2.x, v1.y << v2.y};                                 \
  }                                                                      \
  CHD_FUNC T2 operator>>(T2 const& v, T scalar) {                        \
    return {v.x >> scalar, v.y >> scalar};                               \
  }                                                                      \
  CHD_FUNC T2 operator>>(T scalar, T2 const& v) {                        \
    return {scalar >> v.x, scalar >> v.y};                               \
  }                                                                      \
  CHD_FUNC T2 operator>>(T2 const& v1, T2 const& v2) {                   \
    return {v1.x >> v2.x, v1.y >> v2.y};                                 \
  }                                                                      \
  CHD_FUNC T2 operator~(T2 const& v) {                                   \
    return {~v.x, ~v.y};                                                 \
  }                                                                      \
  CHD_FUNC T3 operator&(T3 const& v, T scalar) {                         \
    return {v.x & scalar, v.y & scalar, v.z & scalar};                   \
  }                                                                      \
  CHD_FUNC T3 operator&(T scalar, T3 const& v) {                         \
    return {scalar & v.x, scalar & v.y, scalar & v.z};                   \
  }                                                                      \
  CHD_FUNC T3 operator&(T3 const& v1, T3 const& v2) {                    \
    return {v1.x & v2.x, v1.y & v2.y, v1.z & v2.z};                      \
  }                                                                      \
  CHD_FUNC T3 operator|(T3 const& v, T scalar) {                         \
    return {v.x | scalar, v.y | scalar, v.z | scalar};                   \
  }                                                                      \
  CHD_FUNC T3 operator|(T scalar, T3 const& v) {                         \
    return {scalar | v.x, scalar | v.y, scalar | v.z};                   \
  }                                                                      \
  CHD_FUNC T3 operator|(T3 const& v1, T3 const& v2) {                    \
    return {v1.x | v2.x, v1.y | v2.y, v1.z | v2.z};                      \
  }                                                                      \
  CHD_FUNC T3 operator^(T3 const& v, T scalar) {                         \
    return {v.x ^ scalar, v.y ^ scalar, v.z ^ scalar};                   \
  }                                                                      \
  CHD_FUNC T3 operator^(T scalar, T3 const& v) {                         \
    return {scalar ^ v.x, scalar ^ v.y, scalar ^ v.z};                   \
  }                                                                      \
  CHD_FUNC T3 operator^(T3 const& v1, T3 const& v2) {                    \
    return {v1.x ^ v2.x, v1.y ^ v2.y, v1.z ^ v2.z};                      \
  }                                                                      \
  CHD_FUNC T3 operator<<(T3 const& v, T scalar) {                        \
    return {v.x << scalar, v.y << scalar, v.z << scalar};                \
  }                                                                      \
  CHD_FUNC T3 operator<<(T scalar, T3 const& v) {                        \
    return {scalar << v.x, scalar << v.y, scalar << v.z};                \
  }                                                                      \
  CHD_FUNC T3 operator<<(T3 const& v1, T3 const& v2) {                   \
    return {v1.x << v2.x, v1.y << v2.y, v1.z << v2.z};                   \
  }                                                                      \
  CHD_FUNC T3 operator>>(T3 const& v, T scalar) {                        \
    return {v.x >> scalar, v.y >> scalar, v.z >> scalar};                \
  }                                                                      \
  CHD_FUNC T3 operator>>(T scalar, T3 const& v) {                        \
    return {scalar >> v.x, scalar >> v.y, scalar >> v.z};                \
  }                                                                      \
  CHD_FUNC T3 operator>>(T3 const& v1, T3 const& v2) {                   \
    return {v1.x >> v2.x, v1.y >> v2.y, v1.z >> v2.z};                   \
  }                                                                      \
  CHD_FUNC T3 operator~(T3 const& v) {                                   \
    return {~v.x, ~v.y, ~v.z};                                           \
  }                                                                      \
  CHD_FUNC T4 operator&(T4 const& v, T scalar) {                         \
    return {v.x & scalar, v.y & scalar, v.z & scalar, v.w & scalar};     \
  }                                                                      \
  CHD_FUNC T4 operator&(T scalar, T4 const& v) {                         \
    return {scalar & v.x, scalar & v.y, scalar & v.z, scalar & v.w};     \
  }                                                                      \
  CHD_FUNC T4 operator&(T4 const& v1, T4 const& v2) {                    \
    return {v1.x & v2.x, v1.y & v2.y, v1.z & v2.z, v1.w & v2.w};         \
  }                                                                      \
  CHD_FUNC T4 operator|(T4 const& v, T scalar) {                         \
    return {v.x | scalar, v.y | scalar, v.z | scalar, v.w | scalar};     \
  }                                                                      \
  CHD_FUNC T4 operator|(T scalar, T4 const& v) {                         \
    return {scalar | v.x, scalar | v.y, scalar | v.z, scalar | v.w};     \
  }                                                                      \
  CHD_FUNC T4 operator|(T4 const& v1, T4 const& v2) {                    \
    return {v1.x | v2.x, v1.y | v2.y, v1.z | v2.z, v1.w | v2.w};         \
  }                                                                      \
  CHD_FUNC T4 operator^(T4 const& v, T scalar) {                         \
    return {v.x ^ scalar, v.y ^ scalar, v.z ^ scalar, v.w ^ scalar};     \
  }                                                                      \
  CHD_FUNC T4 operator^(T scalar, T4 const& v) {                         \
    return {scalar ^ v.x, scalar ^ v.y, scalar ^ v.z, scalar ^ v.w};     \
  }                                                                      \
  CHD_FUNC T4 operator^(T4 const& v1, T4 const& v2) {                    \
    return {v1.x ^ v2.x, v1.y ^ v2.y, v1.z ^ v2.z, v1.w ^ v2.w};         \
  }                                                                      \
  CHD_FUNC T4 operator<<(T4 const& v, T scalar) {                        \
    return {v.x << scalar, v.y << scalar, v.z << scalar, v.w << scalar}; \
  }                                                                      \
  CHD_FUNC T4 operator<<(T scalar, T4 const& v) {                        \
    return {scalar << v.x, scalar << v.y, scalar << v.z, scalar << v.w}; \
  }                                                                      \
  CHD_FUNC T4 operator<<(T4 const& v1, T4 const& v2) {                   \
    return {v1.x << v2.x, v1.y << v2.y, v1.z << v2.z, v1.w << v2.w};     \
  }                                                                      \
  CHD_FUNC T4 operator>>(T4 const& v, T scalar) {                        \
    return {v.x >> scalar, v.y >> scalar, v.z >> scalar, v.w >> scalar}; \
  }                                                                      \
  CHD_FUNC T4 operator>>(T scalar, T4 const& v) {                        \
    return {scalar >> v.x, scalar >> v.y, scalar >> v.z, scalar >> v.w}; \
  }                                                                      \
  CHD_FUNC T4 operator>>(T4 const& v1, T4 const& v2) {                   \
    return {v1.x >> v2.x, v1.y >> v2.y, v1.z >> v2.z, v1.w >> v2.w};     \
  }                                                                      \
  CHD_FUNC T4 operator~(T4 const& v) {                                   \
    return {~v.x, ~v.y, ~v.z, ~v.w};                                     \
  }

#define BINARY_EQ_OP(T, T2, T3, T4)                                      \
  CHD_FUNC bool operator==(T2 const& v1, T2 const& v2) {                 \
    return v1.x == v2.x && v1.y == v2.y;                                 \
  }                                                                      \
  CHD_FUNC bool operator!=(T2 const& v1, T2 const& v2) {                 \
    return !(v1 == v2);                                                  \
  }                                                                      \
  CHD_FUNC bool operator==(T3 const& v1, T3 const& v2) {                 \
    return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;                 \
  }                                                                      \
  CHD_FUNC bool operator!=(T3 const& v1, T3 const& v2) {                 \
    return !(v1 == v2);                                                  \
  }                                                                      \
  CHD_FUNC bool operator==(T4 const& v1, T4 const& v2) {                 \
    return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w; \
  }                                                                      \
  CHD_FUNC bool operator!=(T4 const& v1, T4 const& v2) {                 \
    return !(v1 == v2);                                                  \
  }

// These apply for all types
#define OTHER_FUNC_ALL(T, T2, T3, T4)                                            \
  CHD_FUNC bool all_less(T2 const& v1, T2 const& v2) {                           \
    return (v1.x < v2.x) && (v1.y < v2.y);                                       \
  }                                                                              \
  CHD_FUNC bool all_less_or_eq(T2 const& v1, T2 const& v2) {                     \
    return (v1.x <= v2.x) && (v1.y <= v2.y);                                     \
  }                                                                              \
  CHD_FUNC bool all_greater(T2 const& v1, T2 const& v2) {                        \
    return (v1.x > v2.x) && (v1.y > v2.y);                                       \
  }                                                                              \
  CHD_FUNC bool all_greater_or_eq(T2 const& v1, T2 const& v2) {                  \
    return (v1.x >= v2.x) && (v1.y >= v2.y);                                     \
  }                                                                              \
  CHD_FUNC bool all_less(T3 const& v1, T3 const& v2) {                           \
    return (v1.x < v2.x) && (v1.y < v2.y) && (v1.z < v2.z);                      \
  }                                                                              \
  CHD_FUNC bool all_less_or_eq(T3 const& v1, T3 const& v2) {                     \
    return (v1.x <= v2.x) && (v1.y <= v2.y) && (v1.z <= v2.z);                   \
  }                                                                              \
  CHD_FUNC bool all_greater(T3 const& v1, T3 const& v2) {                        \
    return (v1.x > v2.x) && (v1.y > v2.y) && (v1.z > v2.z);                      \
  }                                                                              \
  CHD_FUNC bool all_greater_or_eq(T3 const& v1, T3 const& v2) {                  \
    return (v1.x >= v2.x) && (v1.y >= v2.y) && (v1.z >= v2.z);                   \
  }                                                                              \
  CHD_FUNC bool all_less(T4 const& v1, T4 const& v2) {                           \
    return (v1.x < v2.x) && (v1.y < v2.y) && (v1.z < v2.z) && (v1.w < v2.w);     \
  }                                                                              \
  CHD_FUNC bool all_less_or_eq(T4 const& v1, T4 const& v2) {                     \
    return (v1.x <= v2.x) && (v1.y <= v2.y) && (v1.z <= v2.z) && (v1.w <= v2.w); \
  }                                                                              \
  CHD_FUNC bool all_greater(T4 const& v1, T4 const& v2) {                        \
    return (v1.x > v2.x) && (v1.y > v2.y) && (v1.z > v2.z) && (v1.w > v2.w);     \
  }                                                                              \
  CHD_FUNC bool all_greater_or_eq(T4 const& v1, T4 const& v2) {                  \
    return (v1.x >= v2.x) && (v1.y >= v2.y) && (v1.z >= v2.z) && (v1.w >= v2.w); \
  }                                                                              \
  CHD_FUNC bool any_less(T2 const& v1, T2 const& v2) {                           \
    return (v1.x < v2.x) || (v1.y < v2.y);                                       \
  }                                                                              \
  CHD_FUNC bool any_less_or_eq(T2 const& v1, T2 const& v2) {                     \
    return (v1.x <= v2.x) || (v1.y <= v2.y);                                     \
  }                                                                              \
  CHD_FUNC bool any_greater(T2 const& v1, T2 const& v2) {                        \
    return (v1.x > v2.x) || (v1.y > v2.y);                                       \
  }                                                                              \
  CHD_FUNC bool any_greater_or_eq(T2 const& v1, T2 const& v2) {                  \
    return (v1.x >= v2.x) || (v1.y >= v2.y);                                     \
  }                                                                              \
  CHD_FUNC bool any_less(T3 const& v1, T3 const& v2) {                           \
    return (v1.x < v2.x) || (v1.y < v2.y) || (v1.z < v2.z);                      \
  }                                                                              \
  CHD_FUNC bool any_less_or_eq(T3 const& v1, T3 const& v2) {                     \
    return (v1.x <= v2.x) || (v1.y <= v2.y) || (v1.z <= v2.z);                   \
  }                                                                              \
  CHD_FUNC bool any_greater(T3 const& v1, T3 const& v2) {                        \
    return (v1.x > v2.x) || (v1.y > v2.y) || (v1.z > v2.z);                      \
  }                                                                              \
  CHD_FUNC bool any_greater_or_eq(T3 const& v1, T3 const& v2) {                  \
    return (v1.x >= v2.x) || (v1.y >= v2.y) || (v1.z >= v2.z);                   \
  }                                                                              \
  CHD_FUNC bool any_less(T4 const& v1, T4 const& v2) {                           \
    return (v1.x < v2.x) || (v1.y < v2.y) || (v1.z < v2.z) || (v1.w < v2.w);     \
  }                                                                              \
  CHD_FUNC bool any_less_or_eq(T4 const& v1, T4 const& v2) {                     \
    return (v1.x <= v2.x) || (v1.y <= v2.y) || (v1.z <= v2.z) || (v1.w <= v2.w); \
  }                                                                              \
  CHD_FUNC bool any_greater(T4 const& v1, T4 const& v2) {                        \
    return (v1.x > v2.x) || (v1.y > v2.y) || (v1.z > v2.z) || (v1.w > v2.w);     \
  }                                                                              \
  CHD_FUNC bool any_greater_or_eq(T4 const& v1, T4 const& v2) {                  \
    return (v1.x >= v2.x) || (v1.y >= v2.y) || (v1.z >= v2.z) || (v1.w >= v2.w); \
  }                                                                              \
  HD_FUNC T2 max(T2 const& v1, T const& v2) {                                    \
    return {max(v1.x, v2), max(v1.y, v2)};                                       \
  }                                                                              \
  HD_FUNC T2 max(T2 const& v1, T2 const& v2) {                                   \
    return {max(v1.x, v2.x), max(v1.y, v2.y)};                                   \
  }                                                                              \
  HD_FUNC T2 min(T2 const& v1, T const& v2) {                                    \
    return {min(v1.x, v2), min(v1.y, v2)};                                       \
  }                                                                              \
  HD_FUNC T2 min(T2 const& v1, T2 const& v2) {                                   \
    return {min(v1.x, v2.x), min(v1.y, v2.y)};                                   \
  }                                                                              \
  HD_FUNC T3 max(T3 const& v1, T const& v2) {                                    \
    return {max(v1.x, v2), max(v1.y, v2), max(v1.z, v2)};                        \
  }                                                                              \
  HD_FUNC T3 max(T3 const& v1, T3 const& v2) {                                   \
    return {max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)};                  \
  }                                                                              \
  HD_FUNC T3 min(T3 const& v1, T const& v2) {                                    \
    return {min(v1.x, v2), min(v1.y, v2), min(v1.z, v2)};                        \
  }                                                                              \
  HD_FUNC T3 min(T3 const& v1, T3 const& v2) {                                   \
    return {min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)};                  \
  }                                                                              \
  HD_FUNC T4 max(T4 const& v1, T const& v2) {                                    \
    return {max(v1.x, v2), max(v1.y, v2), max(v1.z, v2), max(v1.w, v2)};         \
  }                                                                              \
  HD_FUNC T4 max(T4 const& v1, T4 const& v2) {                                   \
    return {max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z), max(v1.w, v2.w)}; \
  }                                                                              \
  HD_FUNC T4 min(T4 const& v1, T const& v2) {                                    \
    return {min(v1.x, v2), min(v1.y, v2), min(v1.z, v2), min(v1.w, v2)};         \
  }                                                                              \
  HD_FUNC T4 min(T4 const& v1, T4 const& v2) {                                   \
    return {min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z), min(v1.w, v2.w)}; \
  }                                                                              \
  HD_FUNC T clamp(T v, T _min, T _max) {                                         \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T2 clamp(T2 v, T2 _min, T2 _max) {                                     \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T3 clamp(T3 v, T3 _min, T3 _max) {                                     \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T4 clamp(T4 v, T4 _min, T4 _max) {                                     \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T2 clamp(T2 v, T _min, T _max) {                                       \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T3 clamp(T3 v, T _min, T _max) {                                       \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  HD_FUNC T4 clamp(T4 v, T _min, T _max) {                                       \
    return min(max(v, _min), _max);                                              \
  }                                                                              \
  CHD_FUNC T mix(T v1, T v2, bool a) {                                           \
    return a ? v2 : v1;                                                          \
  }                                                                              \
  CHD_FUNC T2 mix(T2 v1, T2 v2, bool a) {                                        \
    return a ? v2 : v1;                                                          \
  }                                                                              \
  CHD_FUNC T3 mix(T3 v1, T3 v2, bool a) {                                        \
    return a ? v2 : v1;                                                          \
  }                                                                              \
  CHD_FUNC T4 mix(T4 v1, T4 v2, bool a) {                                        \
    return a ? v2 : v1;                                                          \
  }

// These apply for all types, but unsigned ones
#define ABS_FUNC(T, T2, T3, T4)                      \
  HD_FUNC T2 abs(T2 const& v) {                      \
    return {abs(v.x), abs(v.y)};                     \
  }                                                  \
  HD_FUNC T3 abs(T3 const& v) {                      \
    return {abs(v.x), abs(v.y), abs(v.z)};           \
  }                                                  \
  HD_FUNC T4 abs(T4 const& v) {                      \
    return {abs(v.x), abs(v.y), abs(v.z), abs(v.w)}; \
  }

// Make functions
#define MAKE_FUNC(T, T2, T3, T4)                            \
  HD_FUNC T2 make_##T2(T scalar) {                          \
    return {scalar, scalar};                                \
  }                                                         \
  HD_FUNC T3 make_##T3(T scalar) {                          \
    return {scalar, scalar, scalar};                        \
  }                                                         \
  HD_FUNC T4 make_##T4(T scalar) {                          \
    return {scalar, scalar, scalar, scalar};                \
  }                                                         \
  HD_FUNC T3 make_##T3(T2 const& v, T scalar) {             \
    return {v.x, v.y, scalar};                              \
  }                                                         \
  HD_FUNC T3 make_##T3(T scalar, T2 const& v) {             \
    return {scalar, v.x, v.y};                              \
  }                                                         \
  HD_FUNC T4 make_##T4(T2 const& v1, T2 const& v2) {        \
    return {v1.x, v1.y, v2.x, v2.y};                        \
  }                                                         \
  HD_FUNC T4 make_##T4(T2 const& v, T scalar1, T scalar2) { \
    return {v.x, v.y, scalar1, scalar2};                    \
  }                                                         \
  HD_FUNC T4 make_##T4(T scalar1, T scalar2, T2 const& v) { \
    return {scalar1, scalar2, v.x, v.y};                    \
  }                                                         \
  HD_FUNC T4 make_##T4(T scalar1, T2 const& v, T scalar2) { \
    return {scalar1, v.x, v.y, scalar2};                    \
  }                                                         \
  HD_FUNC T4 make_##T4(T3 const& v, T scalar) {             \
    return {v.x, v.y, v.z, scalar};                         \
  }                                                         \
  HD_FUNC T4 make_##T4(T scalar, T3 const& v) {             \
    return {scalar, v.x, v.y, v.z};                         \
  }                                                         \
  HD_FUNC T2 make_##T2(T3 const& v) {                       \
    return {v.x, v.y};                                      \
  }                                                         \
  HD_FUNC T2 make_##T2(T4 const& v) {                       \
    return {v.x, v.y};                                      \
  }                                                         \
  HD_FUNC T3 make_##T3(T4 const& v) {                       \
    return {v.x, v.y, v.z};                                 \
  }

#define OTHER_FUNC_INT(T, T2, T3, T4)                                                            \
  CHD_FUNC T floor_div(T a, T b) {                                                               \
    T t = 1 - a / b;                                                                             \
    return (a + t * b) / b - t;                                                                  \
  }                                                                                              \
  CHD_FUNC T2 floor_div(T2 const& v1, T2 const& v2) {                                            \
    return {floor_div(v1.x, v2.x), floor_div(v1.y, v2.y)};                                       \
  }                                                                                              \
  CHD_FUNC T2 floor_div(T2 const& v1, T v2) {                                                    \
    return {floor_div(v1.x, v2), floor_div(v1.y, v2)};                                           \
  }                                                                                              \
  CHD_FUNC T3 floor_div(T3 const& v1, T3 const& v2) {                                            \
    return {floor_div(v1.x, v2.x), floor_div(v1.y, v2.y), floor_div(v1.z, v2.z)};                \
  }                                                                                              \
  CHD_FUNC T3 floor_div(T3 const& v1, T v2) {                                                    \
    return {floor_div(v1.x, v2), floor_div(v1.y, v2), floor_div(v1.z, v2)};                      \
  }                                                                                              \
  CHD_FUNC T4 floor_div(T4 const& v1, T4 const& v2) {                                            \
    return {                                                                                     \
        floor_div(v1.x, v2.x),                                                                   \
        floor_div(v1.y, v2.y),                                                                   \
        floor_div(v1.z, v2.z),                                                                   \
        floor_div(v1.w, v2.w)};                                                                  \
  }                                                                                              \
  CHD_FUNC T4 floor_div(T4 const& v1, T v2) {                                                    \
    return {floor_div(v1.x, v2), floor_div(v1.y, v2), floor_div(v1.z, v2), floor_div(v1.w, v2)}; \
  }

#define OTHER_FUNC_FP(T, T2, T3, T4)                                                         \
  CHD_FUNC T dot(T2 a, T2 b) {                                                               \
    return a.x * b.x + a.y * b.y;                                                            \
  }                                                                                          \
  CHD_FUNC T dot(T3 a, T3 b) {                                                               \
    return a.x * b.x + a.y * b.y + a.z * b.z;                                                \
  }                                                                                          \
  CHD_FUNC T dot(T4 a, T4 b) {                                                               \
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                    \
  }                                                                                          \
  CHD_FUNC T cross(T2 a, T2 b) {                                                             \
    return a.x * b.y - a.y * b.x;                                                            \
  }                                                                                          \
  CHD_FUNC T3 cross(T3 a, T3 b) {                                                            \
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};            \
  }                                                                                          \
  HD_FUNC T norm(T2 a) {                                                                     \
    return sqrt(dot(a, a));                                                                  \
  }                                                                                          \
  HD_FUNC T norm(T3 a) {                                                                     \
    return detail::norm3d(a.x, a.y, a.z);                                                    \
  }                                                                                          \
  HD_FUNC T norm(T4 a) {                                                                     \
    return detail::norm4d(a.x, a.y, a.z, a.w);                                               \
  }                                                                                          \
  HD_FUNC T rnorm(T2 a) {                                                                    \
    return rsqrt(dot(a, a));                                                                 \
  }                                                                                          \
  HD_FUNC T rnorm(T3 a) {                                                                    \
    return detail::rnorm3d(a.x, a.y, a.z);                                                   \
  }                                                                                          \
  HD_FUNC T rnorm(T4 a) {                                                                    \
    return detail::rnorm4d(a.x, a.y, a.z, a.w);                                              \
  }                                                                                          \
  HD_FUNC T2 normalize(T2 v) {                                                               \
    T invLen = rnorm(v);                                                                     \
    return v * invLen;                                                                       \
  }                                                                                          \
  HD_FUNC T3 normalize(T3 v) {                                                               \
    T invLen = rnorm(v);                                                                     \
    return v * invLen;                                                                       \
  }                                                                                          \
  HD_FUNC T4 normalize(T4 v) {                                                               \
    T invLen = rnorm(v);                                                                     \
    return v * invLen;                                                                       \
  }                                                                                          \
  HD_FUNC T2 saturate(T2 v) {                                                                \
    return {saturate(v.x), saturate(v.y)};                                                   \
  }                                                                                          \
  HD_FUNC T3 saturate(T3 v) {                                                                \
    return {saturate(v.x), saturate(v.y), saturate(v.z)};                                    \
  }                                                                                          \
  HD_FUNC T4 saturate(T4 v) {                                                                \
    return {saturate(v.x), saturate(v.y), saturate(v.z), saturate(v.w)};                     \
  }                                                                                          \
  CHD_FUNC T sign(T v) {                                                                     \
    return v > 0 ? 1 : (v < 0 ? -1 : 0);                                                     \
  }                                                                                          \
  CHD_FUNC T2 sign(T2 v) {                                                                   \
    return {sign(v.x), sign(v.y)};                                                           \
  }                                                                                          \
  CHD_FUNC T3 sign(T3 v) {                                                                   \
    return {sign(v.x), sign(v.y), sign(v.z)};                                                \
  }                                                                                          \
  CHD_FUNC T4 sign(T4 v) {                                                                   \
    return {sign(v.x), sign(v.y), sign(v.z), sign(v.w)};                                     \
  }                                                                                          \
  CHD_FUNC T mix(T v1, T v2, T a) {                                                          \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T2 mix(T2 v1, T2 v2, T a) {                                                       \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T3 mix(T3 v1, T3 v2, T a) {                                                       \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T4 mix(T4 v1, T4 v2, T a) {                                                       \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T2 mix(T2 v1, T2 v2, T2 a) {                                                      \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T3 mix(T3 v1, T3 v2, T3 a) {                                                      \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T4 mix(T4 v1, T4 v2, T4 a) {                                                      \
    return v1 * (T(1.0) - a) + v2 * a;                                                       \
  }                                                                                          \
  CHD_FUNC T sum(T2 const& v) {                                                              \
    return v.x + v.y;                                                                        \
  }                                                                                          \
  CHD_FUNC T sum(T3 const& v) {                                                              \
    return v.x + v.y + v.z;                                                                  \
  }                                                                                          \
  CHD_FUNC T sum(T4 const& v) {                                                              \
    return v.x + v.y + v.z + v.w;                                                            \
  }                                                                                          \
  HD_FUNC T epsclamp(T v, T eps) {                                                           \
    return (v < 0) ? min(v, -eps) : max(v, eps);                                             \
  }                                                                                          \
  HD_FUNC T epsclamp(T v) {                                                                  \
    return epsclamp(v, epsilon<T>::value);                                                   \
  }                                                                                          \
  HD_FUNC T2 epsclamp(T2 v) {                                                                \
    return {epsclamp(v.x), epsclamp(v.y)};                                                   \
  }                                                                                          \
  HD_FUNC T3 epsclamp(T3 v) {                                                                \
    return {epsclamp(v.x), epsclamp(v.y), epsclamp(v.z)};                                    \
  }                                                                                          \
  HD_FUNC T4 epsclamp(T4 v) {                                                                \
    return {epsclamp(v.x), epsclamp(v.y), epsclamp(v.z), epsclamp(v.w)};                     \
  }                                                                                          \
  HD_FUNC T2 epsclamp(T2 v, T eps) {                                                         \
    return {epsclamp(v.x, eps), epsclamp(v.y, eps)};                                         \
  }                                                                                          \
  HD_FUNC T3 epsclamp(T3 v, T eps) {                                                         \
    return {epsclamp(v.x, eps), epsclamp(v.y, eps), epsclamp(v.z, eps)};                     \
  }                                                                                          \
  HD_FUNC T4 epsclamp(T4 v, T eps) {                                                         \
    return {epsclamp(v.x, eps), epsclamp(v.y, eps), epsclamp(v.z, eps), epsclamp(v.w, eps)}; \
  }                                                                                          \
  CHD_FUNC void inverse(const T2(&m)[2], T2(&out)[2]) {                                      \
    T det_m = T(1.0) / (m[0].x * m[1].y - m[0].y * m[1].x);                                  \
    out[0] = det_m * T2({m[1].y, -m[0].y});                                                  \
    out[1] = det_m * T2({-m[1].x, m[0].x});                                                  \
  }                                                                                          \
  CHD_FUNC void inverse(const T3(&m)[3], T3(&out)[3]) {                                      \
    T det_m = T(1.0) /                                                                       \
        (+m[0].x * (m[1].y * m[2].z - m[1].z * m[2].y) -                                     \
         m[0].y * (m[1].x * m[2].z - m[1].z * m[2].x) +                                      \
         m[0].z * (m[1].x * m[2].y - m[1].y * m[2].x));                                      \
    out[0] = det_m *                                                                         \
        T3({                                                                                 \
            +(m[1].y * m[2].z - m[2].y * m[1].z),                                            \
            -(m[0].y * m[2].z - m[2].y * m[0].z),                                            \
            +(m[0].y * m[1].z - m[1].y * m[0].z),                                            \
        });                                                                                  \
    out[1] = det_m *                                                                         \
        T3({                                                                                 \
            -(m[1].x * m[2].z - m[2].x * m[1].z),                                            \
            +(m[0].x * m[2].z - m[2].x * m[0].z),                                            \
            -(m[0].x * m[1].z - m[1].x * m[0].z),                                            \
        });                                                                                  \
    out[2] = det_m *                                                                         \
        T3({                                                                                 \
            +(m[1].x * m[2].y - m[2].x * m[1].y),                                            \
            -(m[0].x * m[2].y - m[2].x * m[0].y),                                            \
            +(m[0].x * m[1].y - m[1].x * m[0].y),                                            \
        });                                                                                  \
  }                                                                                          \
  CHD_FUNC T2 mul(const T2(&r)[2], T2 v) {                                                   \
    return T2({dot(r[0], v), dot(r[1], v)});                                                 \
  }                                                                                          \
  CHD_FUNC T3 mul(const T3(&r)[3], T3 v) {                                                   \
    return T3({dot(r[0], v), dot(r[1], v), dot(r[2], v)});                                   \
  }                                                                                          \
  CHD_FUNC T4 mul(const T4(&r)[4], T4 v) {                                                   \
    return T4({dot(r[0], v), dot(r[1], v), dot(r[2], v), dot(r[3], v)});                     \
  }                                                                                          \
  CHD_FUNC void mul(const T2(&a)[2], const T2(&b)[2], T2(&out)[2]) {                         \
    out[0] = T2({dot(a[0], T2({b[0].x, b[1].x})), dot(a[0], T2({b[0].y, b[1].y}))});         \
    out[1] = T2({dot(a[1], T2({b[0].x, b[1].x})), dot(a[1], T2({b[0].y, b[1].y}))});         \
  }                                                                                          \
  CHD_FUNC void mul(const T2(&a)[2], const T3(&b)[2], T3(&out)[2]) {                         \
    out[0] =                                                                                 \
        T3({dot(a[0], T2({b[0].x, b[1].x})),                                                 \
            dot(a[0], T2({b[0].y, b[1].y})),                                                 \
            dot(a[0], T2({b[0].z, b[1].z}))});                                               \
    out[1] =                                                                                 \
        T3({dot(a[1], T2({b[0].x, b[1].x})),                                                 \
            dot(a[1], T2({b[0].y, b[1].y})),                                                 \
            dot(a[1], T2({b[0].z, b[1].z}))});                                               \
  }                                                                                          \
  CHD_FUNC void mul(const T3(&a)[3], const T3(&b)[3], T3(&out)[3]) {                         \
    out[0] =                                                                                 \
        T3({dot(a[0], T3({b[0].x, b[1].x, b[2].x})),                                         \
            dot(a[0], T3({b[0].y, b[1].y, b[2].y})),                                         \
            dot(a[0], T3({b[0].z, b[1].z, b[2].z}))});                                       \
    out[1] =                                                                                 \
        T3({dot(a[1], T3({b[0].x, b[1].x, b[2].x})),                                         \
            dot(a[1], T3({b[0].y, b[1].y, b[2].y})),                                         \
            dot(a[1], T3({b[0].z, b[1].z, b[2].z}))});                                       \
    out[2] =                                                                                 \
        T3({dot(a[2], T3({b[0].x, b[1].x, b[2].x})),                                         \
            dot(a[2], T3({b[0].y, b[1].y, b[2].y})),                                         \
            dot(a[2], T3({b[0].z, b[1].z, b[2].z}))});                                       \
  }

#define DEFINE_FUNC_FOR_UNSIGNED_INT(T, T2, T3, T4) \
  UNARY_OP(T, T2, T3, T4)                           \
  BINARY_ARITHM_OP(T, T2, T3, T4)                   \
  BINARY_BIT_OP(T, T2, T3, T4)                      \
  BINARY_EQ_OP(T, T2, T3, T4)                       \
  BINARY_INT_OP(T, T2, T3, T4)                      \
  OTHER_FUNC_ALL(T, T2, T3, T4)                     \
  OTHER_FUNC_INT(T, T2, T3, T4)                     \
  MAKE_FUNC(T, T2, T3, T4)

#define DEFINE_FUNC_FOR_SIGNED_INT(T, T2, T3, T4) \
  DEFINE_FUNC_FOR_UNSIGNED_INT(T, T2, T3, T4)     \
  ABS_FUNC(T, T2, T3, T4)

#define DEFINE_FUNC_FOR_FLOAT(T, T2, T3, T4) \
  UNARY_OP(T, T2, T3, T4)                    \
  BINARY_ARITHM_OP(T, T2, T3, T4)            \
  BINARY_EQ_OP(T, T2, T3, T4)                \
  OTHER_FUNC_ALL(T, T2, T3, T4)              \
  OTHER_FUNC_FP(T, T2, T3, T4)               \
  ABS_FUNC(T, T2, T3, T4)                    \
  MAKE_FUNC(T, T2, T3, T4)

DEFINE_FUNC_FOR_UNSIGNED_INT(unsigned int, uint2, uint3, uint4);
DEFINE_FUNC_FOR_SIGNED_INT(int, int2, int3, int4);
DEFINE_FUNC_FOR_FLOAT(float, float2, float3, float4);
DEFINE_FUNC_FOR_FLOAT(double, double2, double3, double4);

namespace detail {
template <typename scalar_t>
struct VecType;
}

// Type inference utils for writing templates
//
// Derive vector type given the scalar type:
// math::TVec2<float> a; // `a` is of type `float2`
// math::TVec3<int> b; // `b` is of type `int3`;
//
// Derive vector type given the vector size and scalar type:
// math::TVec<double, 4> c; // `c` is of type `double4`;
template <typename scalar_t>
using TVec1 = typename detail::VecType<scalar_t>::scalar1_t;

template <typename scalar_t>
using TVec2 = typename detail::VecType<scalar_t>::scalar2_t;

template <typename scalar_t>
using TVec3 = typename detail::VecType<scalar_t>::scalar3_t;

template <typename scalar_t>
using TVec4 = typename detail::VecType<scalar_t>::scalar4_t;

template <typename scalar_t, int D>
using TVec = typename detail::VecType<scalar_t>::template dim<D>::type;

namespace detail {
template <int D, template <typename scalar_t> class Vec, typename scalar_t>
struct VecD;

template <template <typename scalar_t> class Vec, typename scalar_t>
struct VecD<1, Vec, scalar_t> {
  typedef typename Vec<scalar_t>::scalar1_t type;
};
template <template <typename scalar_t> class Vec, typename scalar_t>
struct VecD<2, Vec, scalar_t> {
  typedef typename Vec<scalar_t>::scalar2_t type;
};
template <template <typename scalar_t> class Vec, typename scalar_t>
struct VecD<3, Vec, scalar_t> {
  typedef typename Vec<scalar_t>::scalar3_t type;
};
template <template <typename scalar_t> class Vec, typename scalar_t>
struct VecD<4, Vec, scalar_t> {
  typedef typename Vec<scalar_t>::scalar4_t type;
};

#define MH_TYPE_DECLARATION(TYPE, NAME)                               \
  template <>                                                         \
  struct VecType<TYPE> {                                              \
    typedef TYPE scalar_t;                                            \
    typedef NAME##1 scalar1_t;                                        \
    typedef NAME##2 scalar2_t;                                        \
    typedef NAME##3 scalar3_t;                                        \
    typedef NAME##4 scalar4_t;                                        \
    template <int D>                                                  \
    struct dim {                                                      \
      typedef typename detail::VecD<D, VecType, scalar_t>::type type; \
    };                                                                \
  };

MH_TYPE_DECLARATION(float, float)
MH_TYPE_DECLARATION(double, double)
MH_TYPE_DECLARATION(char, char)
MH_TYPE_DECLARATION(unsigned char, uchar)
MH_TYPE_DECLARATION(short, short)
MH_TYPE_DECLARATION(unsigned short, ushort)
MH_TYPE_DECLARATION(int, int)
MH_TYPE_DECLARATION(unsigned int, uint)
} // namespace detail
} // namespace math

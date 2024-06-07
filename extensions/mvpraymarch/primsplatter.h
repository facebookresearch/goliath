// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef MVPRAYMARCHER_PRIMSPLATTER_H_
#define MVPRAYMARCHER_PRIMSPLATTER_H_

struct PrimSplatterDataBase {
    typedef PrimSplatterDataBase base;
};

template<
    template<typename> class GridSplatterT=GridSplatterChlast>
struct PrimSplatterTW {
    struct Data : public PrimSplatterDataBase {
        int s_nstride;
        int SD, SH, SW;
        float * shadow;

        __forceinline__ __device__ void n_stride(int n) {
            shadow += n * s_nstride;
        }
    };

    float * shadow_ptr;

    __forceinline__ __device__ void forward(
            const Data & data,
            int k,
            float3 y0,
            float w) {
        shadow_ptr = data.shadow + (k * 2 * data.SD * data.SH * data.SW);
        GridSplatterT<float>::forward(1, data.SD, data.SH, data.SW, 1.f - w, shadow_ptr, y0, false);
    }
};

#endif

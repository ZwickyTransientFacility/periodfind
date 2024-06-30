// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ls.h"

#include <algorithm>

#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

const float TWO_PI = M_PI * 2.0;

//
// Simple LombScargle Function Definitions
//

LombScargle::LombScargle() {}

//
// CUDA Kernels
//

__global__ void LombScargleKernel(const float* times,
                                  const float* mags,
                                  const size_t length,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_period_dts,
                                  const LombScargle params,
                                  float* periodogram) {
    const size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= num_periods || thread_y >= num_period_dts) {
        return;
    }

    // Period and period time derivative
    const float period = periods[thread_x];
    const float period_dt = period_dts[thread_y];

    // Time derivative correction factor.
    const float pdt_corr = (period_dt / period) / 2;

    float mag_cos = 0.0;
    float mag_sin = 0.0;
    float cos_cos = 0.0;
    float cos_sin = 0.0;

    float cos, sin, i_part;

    for (size_t idx = 0; idx < length; idx++) {
        float t = times[idx];
        float mag = mags[idx];

        float t_corr = t - pdt_corr * t * t;
        float folded = fabsf(modff(t_corr / period, &i_part));

        sincosf(TWO_PI * folded, &sin, &cos);

        mag_cos += mag * cos;
        mag_sin += mag * sin;
        cos_cos += cos * cos;
        cos_sin += cos * sin;
    }

    float sin_sin = static_cast<float>(length) - cos_cos;

    float cos_tau, sin_tau;
    sincosf(0.5 * atan2f(2.0 * cos_sin, cos_cos - sin_sin), &sin_tau, &cos_tau);

    float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
    numerator_l *= numerator_l;

    float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
    numerator_r *= numerator_r;

    float denominator_l = cos_tau * cos_tau * cos_cos
                          + 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * sin_sin;

    float denominator_r = cos_tau * cos_tau * sin_sin
                          - 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * cos_cos;

    periodogram[thread_x * num_period_dts + thread_y] =
        0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
}

__global__ void LombScargleKernelBatched(const float* times,
                                         const float* mags,
                                         const size_t* lengths,
                                         const float* periods,
                                         const float* period_dts,
                                         const size_t num_periods,
                                         const size_t num_period_dts,
                                         const size_t num_curves,
                                         const size_t max_length,
                                         const LombScargle params,
                                         float* periodogram) {
    const size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= num_periods || thread_y >= num_period_dts) {
        return;
    }

    //printf("KERNEL: max_length %lu, num_curves %lu, thread_x %lu, thread_y %lu\n", max_length, num_curves, thread_x, thread_y);

    for (size_t curve_idx = 0; curve_idx < num_curves; ++curve_idx) {
        size_t offset = 0;
        for (size_t i = 0; i < curve_idx; ++i) {
            offset += lengths[i];
        }

        //printf("KERNEL: offset %lu, curve_idx %lu\n", offset, curve_idx);

        if (curve_idx >= num_curves) {
            //printf("Illegal access: curve_idx %lu exceeds num_curves %lu\n", curve_idx, num_curves);
            return;
        }

        const float* times_cur = times + offset;
        const float* mags_cur = mags + offset;

        const size_t length = lengths[curve_idx];
        //printf("KERNEL: length %lu\n", length);

        if (length > max_length) {
            //printf("Illegal memory access: length %lu exceeds max_length %lu\n", length, max_length);
            return;
        }

        // Period and period time derivative
        const float period = periods[thread_x];
        const float period_dt = period_dts[thread_y];

        // Time derivative correction factor.
        const float pdt_corr = (period_dt / period) / 2;

        float mag_cos = 0.0, mag_sin = 0.0, cos_cos = 0.0, cos_sin = 0.0;
        float cos, sin, i_part;

        for (size_t idx = 0; idx < length; idx++) {
            if (idx >= max_length) {
                //printf("Illegal memory access: idx %lu exceeds max_length %lu\n", idx, max_length);
                return;
            }

            float t = times_cur[idx];
            float mag = mags_cur[idx];

            float t_corr = t - pdt_corr * t * t;
            float folded = fabsf(modff(t_corr / period, &i_part));

            sincosf(TWO_PI * folded, &sin, &cos);

            mag_cos += mag * cos;
            mag_sin += mag * sin;
            cos_cos += cos * cos;
            cos_sin += cos * sin;
        }

        float sin_sin = static_cast<float>(length) - cos_cos;
        float cos_tau, sin_tau;
        sincosf(0.5 * atan2f(2.0 * cos_sin, cos_cos - sin_sin), &sin_tau, &cos_tau);

        float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
        numerator_l *= numerator_l;

        float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
        numerator_r *= numerator_r;

        float denominator_l = cos_tau * cos_cos + sin_tau * sin_sin + 2 * cos_tau * sin_tau * cos_sin;

        float denominator_r = cos_tau * sin_sin - sin_tau * cos_cos - 2 * cos_tau * sin_tau * cos_sin;

        periodogram[curve_idx * num_periods * num_period_dts + thread_x * num_period_dts + thread_y] =
            0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
    }
}





//
// Wrapper Functions
//

float* LombScargle::DeviceCalcLS(const float* times,
                                 const float* mags,
                                 const size_t length,
                                 const float* periods,
                                 const float* period_dts,
                                 const size_t num_periods,
                                 const size_t num_p_dts) const {
    float* periodogram;
    gpuErrchk(
        cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float)));

    const size_t x_threads = 256;
    const size_t y_threads = 1;
    const size_t x_blocks = ((num_periods + x_threads - 1) / x_threads);
    const size_t y_blocks = ((num_p_dts + y_threads - 1) / y_threads);

    const dim3 block_dim = dim3(x_threads, y_threads);
    const dim3 grid_dim = dim3(x_blocks, y_blocks);

    LombScargleKernel<<<grid_dim, block_dim>>>(times, mags, length, periods,
                                               period_dts, num_periods,
                                               num_p_dts, *this, periodogram);

    return periodogram;
}

void LombScargle::CalcLS(float* times,
                         float* mags,
                         size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* per_out) const {
    CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                  std::vector<size_t>{length}, periods, period_dts, num_periods,
                  num_p_dts, per_out);
}

float* LombScargle::CalcLS(float* times,
                           float* mags,
                           size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    return CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                         std::vector<size_t>{length}, periods, period_dts,
                         num_periods, num_p_dts);
}

void LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                const std::vector<float*>& mags,
                                const std::vector<size_t>& lengths,
                                const float* periods,
                                const float* period_dts,
                                const size_t num_periods,
                                const size_t num_p_dts,
                                float* per_out) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    const size_t num_curves = 1;

    // Copy trial information over
    float* dev_periods;
    float* dev_period_dts;
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float), cudaMemcpyHostToDevice));

    // Intermediate conditional entropy memory
    float* dev_per;
    gpuErrchk(cudaMalloc(&dev_per, num_curves * per_out_size));

    // Kernel launch information
    const size_t x_threads = 256;
    const size_t y_threads = 1;
    const size_t x_blocks = ((num_periods + x_threads - 1) / x_threads);
    const size_t y_blocks = ((num_p_dts + y_threads - 1) / y_threads);
    const dim3 block_dim = dim3(x_threads, y_threads);
    const dim3 grid_dim = dim3(x_blocks, y_blocks);

    // Buffer size (large enough for the longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length * num_curves;
    printf("num_curves: %lu\n", num_curves);
    printf("buffer length (max_length): %lu\n", buffer_length);
    printf("buffer_bytes: %lu\n", buffer_bytes);
    printf("lengths.size(), bytes: %lu\t%lu\n", lengths.size(), lengths.size() * sizeof(size_t));
    printf("times.size(), bytes: %lu\t%lu\n", times.size(), times.size() * sizeof(float*));
    printf("mags.size(), bytes: %lu\t%lu\n", mags.size(), mags.size() * sizeof(float*));
    printf("per_out_size: %lu\n", per_out_size);

    // Allocate for lengths on the device
    size_t* dev_lengths;
    gpuErrchk(cudaMalloc(&dev_lengths, num_curves * sizeof(size_t)));

    float* dev_times_buffer;
    float* dev_mags_buffer;
    gpuErrchk(cudaMalloc(&dev_times_buffer, buffer_bytes));
    gpuErrchk(cudaMalloc(&dev_mags_buffer, buffer_bytes));

    // Zero conditional entropy output
    gpuErrchk(cudaMemset(dev_per, 0, per_out_size));

    for (size_t batch_idx = 0; batch_idx < lengths.size(); batch_idx += num_curves) {
        size_t curve_bytes = 0;
        // Copy light curve into device buffer
        for (size_t i = 0; i < num_curves; i++) {
            if (batch_idx + i < lengths.size()) {
                curve_bytes += lengths[batch_idx + i] * sizeof(float);
            }
        }

        //printf("HOST: Copying %lu bytes for curve %lu\n", curve_bytes, batch_idx);
        gpuErrchk(cudaMemcpy(dev_times_buffer, times[batch_idx], curve_bytes, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_mags_buffer, mags[batch_idx], curve_bytes, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_lengths, &lengths[batch_idx], num_curves * sizeof(size_t), cudaMemcpyHostToDevice));

        LombScargleKernelBatched<<<grid_dim, block_dim>>>(
            dev_times_buffer, dev_mags_buffer, dev_lengths,
            dev_periods, dev_period_dts, num_periods, num_p_dts, num_curves,
            buffer_length, *this, dev_per);

        gpuErrchk(cudaPeekAtLastError());  // Check for kernel launch errors
        gpuErrchk(cudaDeviceSynchronize());  // Ensure kernel execution completes

        // Copy periodogram back to host
        gpuErrchk(cudaMemcpy(&per_out[batch_idx * per_points], dev_per, num_curves * per_out_size, cudaMemcpyDeviceToHost));
    }

    // Free all of the GPU memory
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    gpuErrchk(cudaFree(dev_lengths));  // Free dev_lengths
    gpuErrchk(cudaFree(dev_per));
    gpuErrchk(cudaFree(dev_times_buffer));
    gpuErrchk(cudaFree(dev_mags_buffer));
}



float* LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                  const std::vector<float*>& mags,
                                  const std::vector<size_t>& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    // Allocate the output CE array so we can copy to it.
    float* per_out = (float*)malloc(per_size_total);

    CalcLSBatched(times, mags, lengths, periods, period_dts, num_periods,
                  num_p_dts, per_out);

    return per_out;
}
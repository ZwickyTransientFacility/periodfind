#include "ls.h"

#include <algorithm>

#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

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

    periodogram[thread_y * num_periods + thread_x] =
        0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
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
    cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float));

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

float* LombScargle::CalcLS(const float* times,
                           const float* mags,
                           const size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    // Number of bytes of input data
    const size_t data_bytes = length * sizeof(float);

    // Allocate device pointers
    float* dev_times;
    float* dev_mags;
    float* dev_periods;
    float* dev_period_dts;
    cudaMalloc(&dev_times, data_bytes);
    cudaMalloc(&dev_mags, data_bytes);
    cudaMalloc(&dev_periods, num_periods * sizeof(float));
    cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float));

    // Copy data to device memory
    cudaMemcpy(dev_times, times, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mags, mags, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
               cudaMemcpyHostToDevice);

    float* dev_periodogram =
        DeviceCalcLS(dev_times, dev_mags, length, dev_periods, dev_period_dts,
                     num_periods, num_p_dts);

    const size_t periodogram_size = num_periods * num_p_dts * sizeof(float);
    float* periodogram = (float*)malloc(periodogram_size);
    cudaMemcpy(periodogram, dev_periodogram, periodogram_size,
               cudaMemcpyDeviceToHost);

    cudaFree(dev_periodogram);

    cudaFree(dev_times);
    cudaFree(dev_mags);
    cudaFree(dev_periods);
    cudaFree(dev_period_dts);

    return periodogram;
}

float* LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                  const std::vector<float*>& mags,
                                  const std::vector<size_t>& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const {
    // TODO: Use async memory transferring
    // TODO: Look at ways of batching data transfer.

    // Size of one CE out array, and total CE output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    // Allocate the output CE array so we can copy to it.
    float* per_host = (float*)malloc(per_size_total);

    // Copy trial information over
    float* dev_periods;
    float* dev_period_dts;
    cudaMalloc(&dev_periods, num_periods * sizeof(float));
    cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float));
    cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
               cudaMemcpyHostToDevice);

    // Intermediate conditional entropy memory
    float* dev_per;
    cudaMalloc(&dev_per, per_out_size);

    // Kernel launch information
    const size_t x_threads = 256;
    const size_t y_threads = 1;
    const size_t x_blocks = ((num_periods + x_threads - 1) / x_threads);
    const size_t y_blocks = ((num_p_dts + y_threads - 1) / y_threads);
    const dim3 block_dim = dim3(x_threads, y_threads);
    const dim3 grid_dim = dim3(x_blocks, y_blocks);

    // Buffer size (large enough for longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length;

    float* dev_times_buffer;
    float* dev_mags_buffer;
    cudaMalloc(&dev_times_buffer, buffer_bytes);
    cudaMalloc(&dev_mags_buffer, buffer_bytes);

    for (size_t i = 0; i < lengths.size(); i++) {
        // Copy light curve into device buffer
        const size_t curve_bytes = lengths[i] * sizeof(float);
        cudaMemcpy(dev_times_buffer, times[i], curve_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dev_mags_buffer, mags[i], curve_bytes,
                   cudaMemcpyHostToDevice);

        // Zero conditional entropy output
        cudaMemset(dev_per, 0, per_out_size);

        LombScargleKernel<<<grid_dim, block_dim>>>(
            dev_times_buffer, dev_mags_buffer, lengths[i], dev_periods,
            dev_period_dts, num_periods, num_p_dts, *this, dev_per);

        // Copy periodogram back to host
        cudaMemcpy(&per_host[i * per_points], dev_per, per_out_size,
                   cudaMemcpyDeviceToHost);
    }

    // Free all of the GPU memory
    cudaFree(dev_periods);
    cudaFree(dev_period_dts);
    cudaFree(dev_per);
    cudaFree(dev_times_buffer);
    cudaFree(dev_mags_buffer);

    return per_host;
}

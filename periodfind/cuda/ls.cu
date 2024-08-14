// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ls.h"

#include <algorithm>

#include <chrono>
#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

const float TWO_PI = M_PI * 2.0;

//
// Simple LombScargle Function Definitions
//

LombScargle::LombScargle()
{
}

//
// CUDA Kernels
//

__global__ void LombScargleKernel(const float *times,
								  const float *mags,
								  const size_t length,
								  const float *periods,
								  const float *period_dts,
								  const size_t num_periods,
								  const size_t num_period_dts,
								  float       *periodogram)
{
	const size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t thread_y = threadIdx.y + blockIdx.y * blockDim.y;

	if(thread_x >= num_periods || thread_y >= num_period_dts)
	{
		return;
	}

	// Period and period time derivative
	const float period    = periods[thread_x];
	const float period_dt = period_dts[thread_y];

	// Time derivative correction factor.
	const float pdt_corr = (period_dt / period) / 2;

	float mag_cos = 0.0;
	float mag_sin = 0.0;
	float cos_cos = 0.0;
	float cos_sin = 0.0;

	float cos, sin, i_part;

	for(size_t idx = 0; idx < length; idx++)
	{
		float t   = times[idx];
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

	float denominator_l = cos_tau * cos_tau * cos_cos + 2 * cos_tau * sin_tau * cos_sin + sin_tau * sin_tau * sin_sin;

	float denominator_r = cos_tau * cos_tau * sin_sin - 2 * cos_tau * sin_tau * cos_sin + sin_tau * sin_tau * cos_cos;

	periodogram[thread_x * num_period_dts + thread_y] =
		0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
}

__global__ void LombScargleKernelBatched(const float *__restrict__ times,
										 const float *__restrict__ mags,
										 const size_t *__restrict__ lengths,
										 const float *__restrict__ periods,
										 const float *__restrict__ period_dts,
										 const size_t num_periods,
										 const size_t num_period_dts,
										 const size_t num_batched_curves,
										 float *__restrict__ periodogram)
{
	const size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t thread_y = threadIdx.y + blockIdx.y * blockDim.y;

#pragma unroll
	for(size_t curve_idx = 0; curve_idx < num_batched_curves; curve_idx++)
	{
		if(thread_x >= num_periods || thread_y >= num_period_dts)
		{
			return;
		}

		const size_t length = lengths[curve_idx];

		size_t offset = 0;
		for(size_t i = 0; i < curve_idx; i++)
		{
			offset += lengths[i];
		}

		// Period and period time derivative
		const float period    = periods[thread_x];
		const float period_dt = period_dts[thread_y];

		// Time derivative correction factor.
		const float pdt_corr = (period_dt / period) / 2;

		float mag_cos = 0.0;
		float mag_sin = 0.0;
		float cos_cos = 0.0;
		float cos_sin = 0.0;

		float cos, sin, i_part;

		for(size_t idx = 0; idx < length; idx++)
		{
			float t   = times[idx + offset];
			float mag = mags[idx + offset];

			float t_corr = t - pdt_corr * t * t;
			float folded = fabsf(modff(t_corr / period, &i_part));

			__sincosf(TWO_PI * folded, &sin, &cos);

			mag_cos += mag * cos;
			mag_sin += mag * sin;
			cos_cos += cos * cos;
			cos_sin += cos * sin;
		}

		float sin_sin = static_cast<float>(length) - cos_cos;

		float cos_tau, sin_tau;
		__sincosf(0.5 * atan2f(2.0 * cos_sin, cos_cos - sin_sin), &sin_tau, &cos_tau);

		float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
		float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
		numerator_l *= numerator_l;
		numerator_r *= numerator_r;

		float denominator_l = cos_tau * cos_tau * cos_cos + 2 * cos_tau * sin_tau * cos_sin + sin_tau * sin_tau * sin_sin;
		float denominator_r = cos_tau * cos_tau * sin_sin - 2 * cos_tau * sin_tau * cos_sin + sin_tau * sin_tau * cos_cos;

		periodogram[(curve_idx * num_periods * num_period_dts) + (thread_x * num_period_dts + thread_y)] =
			0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
	}
}

//
// Wrapper Functions
//

float *LombScargle::DeviceCalcLS(const float *times,
								 const float *mags,
								 const size_t length,
								 const float *periods,
								 const float *period_dts,
								 const size_t num_periods,
								 const size_t num_p_dts) const
{
	float *periodogram;
	gpuErrchk(
		cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float)));

	const size_t x_threads = 256;
	const size_t y_threads = 1;
	const size_t x_blocks  = ((num_periods + x_threads - 1) / x_threads);
	const size_t y_blocks  = ((num_p_dts + y_threads - 1) / y_threads);

	const dim3 block_dim = dim3(x_threads, y_threads);
	const dim3 grid_dim  = dim3(x_blocks, y_blocks);

	LombScargleKernel<<<grid_dim, block_dim>>>(times, mags, length, periods,
											   period_dts, num_periods,
											   num_p_dts, periodogram);

	return periodogram;
}

void LombScargle::CalcLS(float       *times,
						 float       *mags,
						 size_t       length,
						 const float *periods,
						 const float *period_dts,
						 const size_t num_periods,
						 const size_t num_p_dts,
						 float       *per_out) const
{
	CalcLSBatched(std::vector<float *>{times}, std::vector<float *>{mags},
				  std::vector<size_t>{length}, periods, period_dts, num_periods,
				  num_p_dts, per_out);
}

float *LombScargle::CalcLS(float       *times,
						   float       *mags,
						   size_t       length,
						   const float *periods,
						   const float *period_dts,
						   const size_t num_periods,
						   const size_t num_p_dts) const
{
	return CalcLSBatched(std::vector<float *>{times}, std::vector<float *>{mags},
						 std::vector<size_t>{length}, periods, period_dts,
						 num_periods, num_p_dts);
}


// TODO: Add a parameter to batch curves
void LombScargle::CalcLSBatched(const std::vector<float *> &times,
								const std::vector<float *> &mags,
								const std::vector<size_t>  &lengths,
								const float *__restrict__ periods,
								const float *__restrict__ period_dts,
								const size_t num_periods,
								const size_t num_p_dts,
								float *__restrict__ per_out) const
{
	const size_t num_batched_curves     = 256;
	size_t       per_points     = num_periods * num_p_dts;
	size_t       per_out_size   = num_batched_curves * per_points * sizeof(float);
	size_t       per_size_total = per_points * sizeof(float) * lengths.size();

	float *dev_periods;
	float *dev_period_dts;
	gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
	gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
	gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float), cudaMemcpyHostToDevice));

	float *dev_per;
	gpuErrchk(cudaMalloc(&dev_per, per_out_size));

	const size_t total_threads_x = num_periods * num_batched_curves;
	const size_t x_threads = 256;
	const size_t y_threads = 1;
	const size_t x_blocks  = (total_threads_x + x_threads - 1) / x_threads;
	const size_t y_blocks  = (num_p_dts + y_threads - 1) / y_threads;
	const dim3   block_dim = dim3(x_threads, y_threads);
	const dim3   grid_dim  = dim3(x_blocks, y_blocks);

	auto         max_length    = std::max_element(lengths.begin(), lengths.end());
	const size_t buffer_length = *max_length;
	const size_t buffer_bytes  = num_batched_curves * buffer_length * sizeof(float);

	float  *dev_times_buffer;
	float  *dev_mags_buffer;
	size_t *dev_lengths_buffer;
	gpuErrchk(cudaMalloc(&dev_times_buffer, buffer_bytes));
	gpuErrchk(cudaMalloc(&dev_mags_buffer, buffer_bytes));
	gpuErrchk(cudaMalloc(&dev_lengths_buffer, num_batched_curves * sizeof(size_t)));

	size_t total_elements = 0;
	for(size_t i = 0; i < lengths.size(); i++)
	{
		total_elements += lengths[i];
	}

	float *__restrict__ host_times_contiguous;
	float *__restrict__ host_mags_contiguous;
	gpuErrchk(cudaHostAlloc((void **) &host_times_contiguous, total_elements * sizeof(float), cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void **) &host_mags_contiguous, total_elements * sizeof(float), cudaHostAllocDefault));

	size_t contiguous_offset = 0;


	for(size_t i = 0; i < lengths.size(); i++)
	{
		memcpy(host_times_contiguous + contiguous_offset, times[i], lengths[i] * sizeof(float));
		memcpy(host_mags_contiguous + contiguous_offset, mags[i], lengths[i] * sizeof(float));
		contiguous_offset += lengths[i];
	}

	const size_t num_streams = 4;
	cudaStream_t streams[num_streams];
	for(size_t i = 0; i < num_streams; i++)
	{
		gpuErrchk(cudaStreamCreate(&streams[i]));
	}

	size_t curve_offset = 0;

#pragma unroll
	for(size_t batch_idx = 0; batch_idx < lengths.size(); batch_idx += num_batched_curves * num_streams)
	{
		for(size_t stream_idx = 0; stream_idx < num_streams; stream_idx++)
		{
			const size_t stream_batch_idx = batch_idx + stream_idx * num_batched_curves;
			if(stream_batch_idx >= lengths.size())
			{
				break;
			}

			size_t curve_bytes           = 0;
			const size_t num_curves_to_process = stream_batch_idx + num_batched_curves < lengths.size() ? num_batched_curves : lengths.size() - stream_batch_idx;
			const size_t actual_per_out_size   = num_curves_to_process * per_points * sizeof(float);

			for(size_t i = 0; i < num_curves_to_process && stream_batch_idx + i < lengths.size(); i++)
			{
				curve_bytes += lengths[stream_batch_idx + i] * sizeof(float);
			}

			gpuErrchk(cudaMemcpyAsync(dev_times_buffer, host_times_contiguous + curve_offset, curve_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));
			gpuErrchk(cudaMemcpyAsync(dev_mags_buffer, host_mags_contiguous + curve_offset, curve_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));
			gpuErrchk(cudaMemcpyAsync(dev_lengths_buffer, &lengths[stream_batch_idx], num_curves_to_process * sizeof(size_t), cudaMemcpyHostToDevice, streams[stream_idx]));

			LombScargleKernelBatched<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
				dev_times_buffer,
				dev_mags_buffer,
				dev_lengths_buffer,
				dev_periods,
				dev_period_dts,
				num_periods,
				num_p_dts,
				num_curves_to_process,
				dev_per);

			gpuErrchk(cudaMemcpyAsync(&per_out[stream_batch_idx * per_points], dev_per, actual_per_out_size, cudaMemcpyDeviceToHost, streams[stream_idx]));

			curve_offset += curve_bytes / sizeof(float);
		}
	}

	for(size_t i = 0; i < num_streams; ++i)
	{
		gpuErrchk(cudaStreamSynchronize(streams[i]));
		gpuErrchk(cudaStreamDestroy(streams[i]));
	}

	cudaFreeHost(host_times_contiguous);
	cudaFreeHost(host_mags_contiguous);
	gpuErrchk(cudaFree(dev_periods));
	gpuErrchk(cudaFree(dev_period_dts));
	gpuErrchk(cudaFree(dev_per));
	gpuErrchk(cudaFree(dev_lengths_buffer));
	gpuErrchk(cudaFree(dev_times_buffer));
	gpuErrchk(cudaFree(dev_mags_buffer));
}


float *LombScargle::CalcLSBatched(const std::vector<float *> &times,
								  const std::vector<float *> &mags,
								  const std::vector<size_t>  &lengths,
								  const float                *periods,
								  const float                *period_dts,
								  const size_t                num_periods,
								  const size_t                num_p_dts) const
{
	// Size of one periodogram out array, and total periodogram output size.
	size_t per_points     = num_periods * num_p_dts;
	size_t per_out_size   = per_points * sizeof(float);
	size_t per_size_total = per_out_size * lengths.size();

	// Allocate the output CE array so we can copy to it.
	float *per_out = (float *) malloc(per_size_total);

	CalcLSBatched(times, mags, lengths, periods, period_dts, num_periods,
				  num_p_dts, per_out);

	return per_out;
}

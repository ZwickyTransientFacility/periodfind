// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ce.h"

#include <algorithm>
#include <iostream>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple ConditionalEntropy Function Definitions
//

ConditionalEntropy::ConditionalEntropy(size_t n_phase,
									   size_t n_mag,
									   size_t p_overlap,
									   size_t m_overlap)
{
	// Just set number of bins
	num_phase_bins = n_phase;
	num_mag_bins   = n_mag;

	// Just set the overlap
	num_phase_overlap = p_overlap;
	num_mag_overlap   = m_overlap;

	// Calculate bin size accordingly
	phase_bin_size = 1.0 / static_cast<float>(n_phase);
	mag_bin_size   = 1.0 / static_cast<float>(n_mag);
}

__host__ __device__ size_t ConditionalEntropy::NumBins() const
{
	return num_phase_bins * num_mag_bins;
}

__host__ __device__ size_t ConditionalEntropy::NumPhaseBins() const
{
	return num_phase_bins;
}

__host__ __device__ size_t ConditionalEntropy::NumMagBins() const
{
	return num_mag_bins;
}

__host__ __device__ size_t ConditionalEntropy::NumPhaseBinOverlap() const
{
	return num_phase_overlap;
}

__host__ __device__ size_t ConditionalEntropy::NumMagBinOverlap() const
{
	return num_mag_overlap;
}

__host__ __device__ float ConditionalEntropy::PhaseBinSize() const
{
	return phase_bin_size;
}

__host__ __device__ float ConditionalEntropy::MagBinSize() const
{
	return mag_bin_size;
}

__host__ __device__ size_t ConditionalEntropy::PhaseBin(float phase_val) const
{
	return static_cast<size_t>(phase_val / phase_bin_size);
}

__host__ __device__ size_t ConditionalEntropy::MagBin(float mag_val) const
{
	return static_cast<size_t>(mag_val / mag_bin_size);
}

__host__ __device__ size_t ConditionalEntropy::BinIndex(size_t phase_bin,
														size_t mag_bin) const
{
	return phase_bin * num_mag_bins + mag_bin;
}

//
// CUDA Kernels
//

/**
 * Folds and bins the input data across all trial periods and time derivatives.
 *
 * This kernel takes in a time-series of paired times and magnitudes, folding
 * the times according to the given trial periods and time derivatives,
 * outputting a series of histograms into global memory.
 *
 * Each block computes a histogram of the full data series for a given period
 * and period time derivative. As such, the x-dimension of the grid should match
 * the number of trial periods, and the y-dimension of the grid should match the
 * number of trial period time derivatives.
 *
 * Internally, the kernel uses shared memory atomics with a 32-bit integer based
 * histogram, which requires a total of 4 * Histogram Size bytes of shared
 * memory. Due to the use of shared atomics, this kernel will perform poorly on
 * pre-Maxwell GPUs.
 *
 * Note: All arrays must be device-allocated
 *
 * @param times light curve datapoint times
 * @param mags light curve datapoint magnitudes
 * @param periods list of trial periods
 * @param period_dts list of trial period time derivatives
 * @param h_params histogram parameters
 * @param hists array of output histograms
 */
__global__ void FoldBinKernel(const float *__restrict__ times,
							  const float *__restrict__ mags,
							  const size_t length,
							  const float *__restrict__ periods,
							  const float *__restrict__ period_dts,
							  const ConditionalEntropy h_params,
							  float *__restrict__ hists)
{
	// Histogram which this block will produce.
	const size_t block_id          = blockIdx.x * gridDim.y + blockIdx.y;
	float *__restrict__ block_hist = &hists[h_params.NumBins() * block_id];

	// Period and period time derivative for this block.
	const float period    = periods[blockIdx.x];
	const float period_dt = period_dts[blockIdx.y];

	// Time derivative correction factor.
	const float pdt_corr = (period_dt / period) / 2;

	// Shared memory histogram for this thread.
	extern __shared__ uint32_t sh_hist[];

	// Zero the shared memory for this block
	for(size_t i = threadIdx.x; i < h_params.NumBins(); i += blockDim.x)
	{
		sh_hist[i] = 0;
	}

	__syncthreads();

	float i_part; // Only used for modff.

	// Accumulate into this thread's histogram (as many points as needed),
	// simultaneously computing the folded time value
	for(size_t idx = threadIdx.x; idx < length; idx += blockDim.x)
	{
		float t      = times[idx];
		float t_corr = t - pdt_corr * t * t;
		float folded = fabsf(modff(t_corr / period, &i_part));

		size_t phase_bin = h_params.PhaseBin(folded);
		size_t mag_bin   = h_params.MagBin(mags[idx]);

		for(size_t i = 0; i < h_params.NumPhaseBinOverlap(); i++)
		{
			for(size_t j = 0; j < h_params.NumMagBinOverlap(); j++)
			{
				size_t idx =
					h_params.BinIndex((phase_bin + i) % h_params.NumPhaseBins(), (mag_bin + j) % h_params.NumMagBins());
				atomicAdd(&sh_hist[idx], 1);
			}
		}
	}

	__syncthreads();

	size_t div = length * h_params.NumPhaseBinOverlap() * h_params.NumMagBinOverlap();

	// Copy the block's histogram into global memory
	for(size_t i = threadIdx.x; i < h_params.NumBins(); i += blockDim.x)
	{
		block_hist[i] = static_cast<float>(sh_hist[i]) / static_cast<float>(div);
	}
}

/**
 * Computes the conditional entropy for the input histograms.
 *
 * This kernel takes in an arbitrarily long list of histograms with a given set
 * of parameters and computes the conditional entropy for each histogram,
 * outputting a series of values into an array.
 *
 * Internally, each thread is responsible for first computing the conditional
 * entropy of one phase bin of the input (disregarding histogram boundaries),
 * then the values for each thread are accumulated directly into global memory
 * to avoid potential inter-block conflicts. This copies the histogram values
 * relevant for each thread into local registers to provide fast access.
 *
 * TODO: Reimplementing shared memory with proper reductions to avoid atomic
 * accesses might work faster in the end than using the local registers + atomic.
 *
 * Note: All arrays must be device-allocated
 *
 * @param hists array of input histograms
 * @param num_hists number of histograms
 * @param h_params histogram parameters
 * @param ce_vals output array of conditional entropy values
 */
__global__ void ConditionalEntropyKernel(const float *__restrict__ hists,
										 const size_t             num_hists,
										 const ConditionalEntropy h_params,
										 float *__restrict__ ce_vals)
{
	// Which histogram row this thread is summing
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Don't compute for out-of-bounds histograms
	if(idx / h_params.NumPhaseBins() >= num_hists)
	{
		return;
	}

	// Index in the histogram array corresponding to the start of this row
	const size_t offset = idx * h_params.NumMagBins();

	// This works if NumMagBins is fairly small.
	float  local_sum    = 0.0f;
	size_t num_mag_bins = h_params.NumMagBins();
	float  local_hists[128]; // assumes no more than 128 bins. Can't declare with num_mag_bins because c++ enforces known array sizes.
	for(size_t i = 0; i < num_mag_bins; i++)
	{
		local_hists[i] = hists[i + offset];
		local_sum += local_hists[i];
	}

	// Compute per-phase-bin conditional entropy
	// TODO: remove use of global mem?
	float p_j = local_sum; // Store p_j
	local_sum = 0.0f;
	for(size_t i = 0; i < num_mag_bins; i++)
	{
		float p_ij = local_hists[i];
		if(p_ij != 0)
		{
			local_sum += p_ij * logf(p_j / p_ij);
		}
	}

	// Accumulate per-phase-bin conditional entropy into total conditional
	// entropy for the histogram.
	// TODO: Replace with shared memory reduction of some kind. *yikes*
	size_t ce_idx = idx / h_params.NumPhaseBins();
	atomicAdd(&ce_vals[ce_idx], local_sum);
}

//
// Wrapper Functions
//

float *ConditionalEntropy::DeviceFoldAndBin(const float *times,
											const float *mags,
											const size_t length,
											const float *periods,
											const float *period_dts,
											const size_t num_periods,
											const size_t num_p_dts) const
{
	// Number of bytes of global memory required to store output
	size_t bytes = NumBins() * sizeof(float) * num_periods * num_p_dts;

	// Allocate and zero global memory for output histograms
	float *dev_hists;
	gpuErrchk(cudaMalloc(&dev_hists, bytes));
	gpuErrchk(cudaMemset(dev_hists, 0, bytes));

	// Number of threads and corresponding shared memory usage
	const size_t num_threads  = 512;
	const size_t shared_bytes = NumBins() * sizeof(uint32_t);

	// Grid to search over periods and time derivatives
	const dim3 grid_dim = dim3(num_periods, num_p_dts);

	// NOTE: A ConditionalEntropy object is small enough that we can pass it in
	//       the registers by dereferencing it.
	FoldBinKernel<<<grid_dim, num_threads, shared_bytes>>>(
		times, mags, length, periods, period_dts, *this, dev_hists);

	return dev_hists;
}

float *ConditionalEntropy::FoldAndBin(const float *times,
									  const float *mags,
									  const size_t length,
									  const float *periods,
									  const float *period_dts,
									  const size_t num_periods,
									  const size_t num_p_dts) const
{
	// Number of bytes of input data
	const size_t data_bytes = length * sizeof(float);

	// Allocate device pointers
	float *dev_times;
	float *dev_mags;
	float *dev_periods;
	float *dev_period_dts;
	gpuErrchk(cudaMalloc(&dev_times, data_bytes));
	gpuErrchk(cudaMalloc(&dev_mags, data_bytes));
	gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
	gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));

	// Copy data to device memory
	gpuErrchk(cudaMemcpy(dev_times, times, data_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_mags, mags, data_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
						 cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_periods, period_dts, num_p_dts * sizeof(float),
						 cudaMemcpyHostToDevice));

	float *dev_hists =
		DeviceFoldAndBin(dev_times, dev_mags, length, dev_periods,
						 dev_period_dts, num_periods, num_p_dts);

	// Allocate host histograms and copy from device
	size_t bytes = NumBins() * num_periods * num_p_dts * sizeof(float);
	float *hists = (float *) malloc(bytes);
	gpuErrchk(cudaMemcpy(hists, dev_hists, bytes, cudaMemcpyDeviceToHost));

	// Free GPU memory
	gpuErrchk(cudaFree(dev_times));
	gpuErrchk(cudaFree(dev_mags));
	gpuErrchk(cudaFree(dev_periods));
	gpuErrchk(cudaFree(dev_period_dts));
	gpuErrchk(cudaFree(dev_hists));

	return hists;
}

float *ConditionalEntropy::DeviceCalcCEFromHists(const float *hists,
												 const size_t num_hists) const
{
	// Allocate global memory for output conditional entropy values
	float *dev_ces;
	gpuErrchk(cudaMalloc(&dev_ces, num_hists * sizeof(float)));

	const size_t n_t = 512;
	const size_t n_b = ((num_hists * NumPhaseBins()) / n_t) + 1;

	// NOTE: A ConditionalEntropy object is small enough that we can pass it in
	//       the registers by dereferencing it.
	ConditionalEntropyKernel<<<n_b, n_t, n_t * sizeof(float)>>>(
		hists, num_hists, *this, dev_ces);

	return dev_ces;
}

float *ConditionalEntropy::CalcCEFromHists(const float *hists,
										   const size_t num_hists) const
{
	// Number of bytes in the histogram
	const size_t bytes = num_hists * NumBins() * sizeof(float);

	// Allocate device memory for histograms and copy over
	float *dev_hists;
	gpuErrchk(cudaMalloc(&dev_hists, bytes));
	gpuErrchk(cudaMemcpy(dev_hists, hists, bytes, cudaMemcpyHostToDevice));

	float *dev_ces = DeviceCalcCEFromHists(dev_hists, num_hists);

	// Copy CEs to host
	float *ces = (float *) malloc(num_hists * sizeof(float));
	gpuErrchk(cudaMemcpy(ces, dev_ces, num_hists * sizeof(float),
						 cudaMemcpyDeviceToHost));

	// Free GPU memory
	gpuErrchk(cudaFree(dev_hists));
	gpuErrchk(cudaFree(dev_ces));

	return ces;
}

void ConditionalEntropy::CalcCEVals(float       *times,
									float       *mags,
									size_t       length,
									const float *periods,
									const float *period_dts,
									const size_t num_periods,
									const size_t num_p_dts,
									float       *ce_out) const
{
	CalcCEValsBatched(std::vector<float *>{times}, std::vector<float *>{mags},
					  std::vector<size_t>{length}, periods, period_dts,
					  num_periods, num_p_dts, ce_out);
}

float *ConditionalEntropy::CalcCEVals(float       *times,
									  float       *mags,
									  size_t       length,
									  const float *periods,
									  const float *period_dts,
									  const size_t num_periods,
									  const size_t num_p_dts) const
{
	return CalcCEValsBatched(std::vector<float *>{times},
							 std::vector<float *>{mags},
							 std::vector<size_t>{length}, periods, period_dts,
							 num_periods, num_p_dts);
}

void ConditionalEntropy::CalcCEValsBatched(const std::vector<float *> &times,
										   const std::vector<float *> &mags,
										   const std::vector<size_t>  &lengths,
										   const float *__restrict__ periods,
										   const float *__restrict__ period_dts,
										   const size_t num_periods,
										   const size_t num_p_dts,
										   float *__restrict__ ce_out) const
{
	// Size of one CE out array, and total CE output size.
	size_t ce_out_size   = num_periods * num_p_dts * sizeof(float);
	size_t ce_size_total = ce_out_size * lengths.size();

	// Copy trial information over
	float *dev_periods;
	float *dev_period_dts;
	gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
	gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
	gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float), cudaMemcpyHostToDevice));

	// Intermediate histogram memory
	size_t num_hists  = num_periods * num_p_dts;
	size_t hist_bytes = NumBins() * sizeof(float) * num_hists;
	float *dev_hists;
	gpuErrchk(cudaMalloc(&dev_hists, hist_bytes));

	// Intermediate conditional entropy memory
	float *dev_ces;
	gpuErrchk(cudaMalloc(&dev_ces, ce_out_size));

	// Kernel launch information for the fold & bin step
	const size_t num_threads_fb  = 256;
	const size_t shared_bytes_fb = NumBins() * sizeof(uint32_t);
	const dim3   grid_dim_fb     = dim3(num_periods, num_p_dts);

	// Kernel launch information for the ce calculation step
	const size_t num_threads_ce = 256;
	const size_t num_blocks_ce  = ((num_hists * NumPhaseBins()) / num_threads_ce) + 1;

	// Buffer size (large enough for longest light curve)
	auto         max_length    = std::max_element(lengths.begin(), lengths.end());
	const size_t buffer_length = *max_length;
	const size_t buffer_bytes  = sizeof(float) * buffer_length;

	float *dev_times_buffer;
	float *dev_mags_buffer;
	gpuErrchk(cudaMalloc(&dev_times_buffer, buffer_bytes));
	gpuErrchk(cudaMalloc(&dev_mags_buffer, buffer_bytes));

	size_t total_elements = 0;
	for(size_t i = 0; i < lengths.size(); i++)
	{
		total_elements += lengths[i];
	}

	size_t contiguous_offset = 0;
	float *__restrict__ host_times_contiguous;
	float *__restrict__ host_mags_contiguous;
	gpuErrchk(cudaHostAlloc((void **) &host_times_contiguous, total_elements * sizeof(float), cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void **) &host_mags_contiguous, total_elements * sizeof(float), cudaHostAllocDefault));

	for(size_t i = 0; i < lengths.size(); i++)
	{
		memcpy(host_times_contiguous + contiguous_offset, times[i], lengths[i] * sizeof(float));
		memcpy(host_mags_contiguous + contiguous_offset, mags[i], lengths[i] * sizeof(float));
		contiguous_offset += lengths[i];
	}

	const size_t num_streams = 3;
	cudaStream_t streams[num_streams];
	for(size_t i = 0; i < num_streams; i++)
	{
		gpuErrchk(cudaStreamCreate(&streams[i]));
	}

	size_t curve_offset = 0;
	for(size_t batch_idx = 0; batch_idx < lengths.size(); batch_idx += num_streams)
	{
		for(size_t stream_idx = 0; stream_idx < num_streams; stream_idx++)
		{
			const size_t stream_batch_idx = batch_idx + stream_idx;

			if(stream_batch_idx >= lengths.size())
			{
				break;
			}

			const size_t curve_bytes = lengths[stream_batch_idx] * sizeof(float);
			gpuErrchk(cudaMemcpyAsync(dev_times_buffer, host_times_contiguous + curve_offset, curve_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));
			gpuErrchk(cudaMemcpyAsync(dev_mags_buffer, host_mags_contiguous + curve_offset, curve_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));

			gpuErrchk(cudaMemsetAsync(dev_ces, 0, ce_out_size, streams[stream_idx]));

			FoldBinKernel<<<grid_dim_fb, num_threads_fb, shared_bytes_fb, streams[stream_idx]>>>(
				dev_times_buffer, dev_mags_buffer, lengths[stream_batch_idx], dev_periods,
				dev_period_dts, *this, dev_hists);

			ConditionalEntropyKernel<<<num_blocks_ce, num_threads_ce, 0, streams[stream_idx]>>>(
				dev_hists, num_hists, *this, dev_ces);

			gpuErrchk(cudaMemcpyAsync(&ce_out[stream_batch_idx * num_hists], dev_ces, ce_out_size, cudaMemcpyDeviceToHost, streams[stream_idx]));

			curve_offset += curve_bytes / sizeof(float);
		}
	}

	for(size_t i = 0; i < num_streams; ++i)
	{
		gpuErrchk(cudaStreamSynchronize(streams[i]));
		gpuErrchk(cudaStreamDestroy(streams[i]));
	}

	gpuErrchk(cudaFree(dev_periods));
	gpuErrchk(cudaFree(dev_period_dts));
	gpuErrchk(cudaFree(dev_hists));
	gpuErrchk(cudaFree(dev_ces));
	gpuErrchk(cudaFree(dev_times_buffer));
	gpuErrchk(cudaFree(dev_mags_buffer));
	cudaFreeHost(host_times_contiguous);
	cudaFreeHost(host_mags_contiguous);
}


float *ConditionalEntropy::CalcCEValsBatched(const std::vector<float *> &times,
											 const std::vector<float *> &mags,
											 const std::vector<size_t>  &lengths,
											 const float                *periods,
											 const float                *period_dts,
											 const size_t                num_periods,
											 const size_t                num_p_dts) const
{
	// Size of one CE out array, and total CE output size.
	size_t ce_out_size   = num_periods * num_p_dts * sizeof(float);
	size_t ce_size_total = ce_out_size * lengths.size();

	// Allocate host memory for output CE values.
	float *ce_out = (float *) malloc(ce_size_total);

	// Perform CE calculation.
	CalcCEValsBatched(times, mags, lengths, periods, period_dts, num_periods,
					  num_p_dts, ce_out);

	return ce_out;
}

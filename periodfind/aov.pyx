#cython: language_level=3

import numpy as np
from periodfind import Statistics, Periodogram

cimport numpy as np
from libc.stddef cimport size_t
from libcpp.vector cimport vector

# Include numpy <-> c array interop
np.import_array()

cdef extern from "./cuda/aov.h":
    cdef cppclass CppAOV "AOV":
        CppAOV(size_t num_phase,
               size_t num_phase_overlap)
        
        float* CalcAOVVals(const float* times,
                           const float* mags,
                           const size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const
        
        float* CalcAOVValsBatched(const vector[float*]& times,
                                  const vector[float*]& mags,
                                  const vector[size_t]& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const;

cdef class AOV:
    cdef CppAOV* aov

    def __cinit__(self,
                  n_phase=10,
                  n_phase_overlap=1):
        self.aov = new CppAOV(n_phase, n_phase_overlap)
    
    def calc_one(self,
                 np.ndarray[ndim=1, dtype=np.float32_t] times not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] mags not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] periods not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] period_dts not None):
        d_len = len(times)
        n_per = len(periods)
        n_pdt = len(period_dts)
        cdef float* aovs = \
            self.aov.CalcAOVVals(&times[0], &mags[0], d_len, &periods[0], &period_dts[0], n_per, n_pdt)
        cdef np.npy_intp dim[2]
        dim[0] = n_per
        dim[1] = n_pdt
        cdef np.ndarray[ndim=2, dtype=np.float32_t] ces_ndarr = \
            np.PyArray_SimpleNewFromData(2, dim, np.NPY_FLOAT, aovs)
        return ces_ndarr

    def calc(self,
             list times,
             list mags,
             np.ndarray[ndim=1, dtype=np.float32_t] periods,
             np.ndarray[ndim=1, dtype=np.float32_t] period_dts,
             output="stats",
             normalize=False):
        
        # Make sure the number of times and mags matches 
        if len(times) != len(mags):
            return np.zeros([0, 0, 0], dtype=np.float32)
        
        cdef np.ndarray[ndim=1, dtype=np.float32_t] time_arr
        cdef vector[float*] times_ptrs
        cdef vector[size_t] times_lens
        for time_obj in times:
            time_arr = time_obj
            times_ptrs.push_back(&time_arr[0])
            times_lens.push_back(len(time_arr))

        mags_use = []
        if normalize:
            for mag in mags:
                min_v = np.min(mag)
                max_v = np.max(mag)
                scaled = ((mag - min_v) / (max_v - min_v)) * 0.999 + 5e-4
                mags_use.append(scaled)
        else:
            mags_use = mags

        cdef np.ndarray[ndim=1, dtype=np.float32_t] mag_arr
        cdef vector[float*] mags_ptrs
        cdef vector[size_t] mags_lens
        for mag_obj in mags_use:
            mag_arr = mag_obj
            mags_ptrs.push_back(&mag_arr[0])
            mags_lens.push_back(len(mag_arr))

        # Make sure the individual lengths match
        if any(t != m for t, m in zip(times_lens, mags_lens)):
            return np.zeros([0, 0, 0], dtype=np.float32)

        n_per = len(periods)
        n_pdt = len(period_dts)

        cdef float* aovs = self.aov.CalcAOVValsBatched(
            times_ptrs, mags_ptrs, times_lens,
            &periods[0], &period_dts[0], n_per, n_pdt,
        )

        cdef np.npy_intp dim[3]
        dim[0] = len(times)
        dim[1] = n_per
        dim[2] = n_pdt

        cdef np.ndarray[ndim=3, dtype=np.float32_t] aovs_ndarr = \
            np.PyArray_SimpleNewFromData(3, dim, np.NPY_FLOAT, aovs)
        
        if output == 'stats':
            axis = (1, 2)
            means = np.mean(aovs_ndarr, axis=axis, dtype=np.float64)
            stds = np.std(aovs_ndarr, axis=axis, dtype=np.float64)

            all_stats = []
            for i in range(len(times)):
                argmin = np.unravel_index(
                    np.argmin(aovs_ndarr[i]),
                    aovs_ndarr[i].shape,
                )

                argmax = np.unravel_index(
                    np.argmax(aovs_ndarr[i]),
                    aovs_ndarr[i].shape,
                )

                stats = Statistics(
                    [periods[argmax[0]], period_dts[argmax[1]]],
                    means[i],
                    aovs_ndarr[i][argmin],
                    aovs_ndarr[i][argmax],
                    stds[i],
                    True,
                )

                all_stats.append(stats)
            
            return all_stats
        elif output == 'periodogram':
            return [Periodogram(data, [periods, period_dts], True)
                    for data in aovs_ndarr]
        else:
            raise NotImplementedError('Only "stats" output is implemented')

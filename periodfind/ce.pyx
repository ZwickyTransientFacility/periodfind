#cython: language_level=3

import numpy as np
from periodfind import Statistics

cimport numpy as np
from libc.stddef cimport size_t
from libcpp.vector cimport vector

# Include numpy <-> c array interop
np.import_array()

cdef extern from "./cuda/ce.h":
    cdef cppclass CppConditionalEntropy "ConditionalEntropy":
        CppConditionalEntropy(size_t num_phase,
                        size_t num_mag,
                        size_t num_phase_overlap,
                        size_t num_mag_overlap)
        
        float* CalcCEVals(const float* times,
                          const float* mags,
                          const size_t length,
                          const float* periods,
                          const float* period_dts,
                          const size_t num_periods,
                          const size_t num_p_dts) const
        
        float* CalcCEValsBatched(const vector[float*]& times,
                                 const vector[float*]& mags,
                                 const vector[size_t]& lengths,
                                 const float* periods,
                                 const float* period_dts,
                                 const size_t num_periods,
                                 const size_t num_p_dts) const;

cdef class ConditionalEntropy:
    cdef CppConditionalEntropy* ce

    def __cinit__(self,
                  n_phase=10,
                  n_mag=10,
                  n_phase_overlap=0,
                  n_mag_overlap=0):
        self.ce = new CppConditionalEntropy(
            n_phase,
            n_mag,
            n_phase_overlap,
            n_mag_overlap)
    
    def calc_one(self,
                 np.ndarray[ndim=1, dtype=np.float32_t] times not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] mags not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] periods not None,
                 np.ndarray[ndim=1, dtype=np.float32_t] period_dts not None):
        d_len = len(times)
        n_per = len(periods)
        n_pdt = len(period_dts)
        cdef float* ces = \
            self.ce.CalcCEVals(&times[0], &mags[0], d_len, &periods[0], &period_dts[0], n_per, n_pdt)
        cdef np.npy_intp dim[2]
        dim[0] = n_per
        dim[1] = n_pdt
        cdef np.ndarray[ndim=2, dtype=np.float32_t] ces_ndarr = \
            np.PyArray_SimpleNewFromData(2, dim, np.NPY_FLOAT, ces)
        return ces_ndarr

    def calc(self,
             list times,
             list mags,
             np.ndarray[ndim=1, dtype=np.float32_t] periods,
             np.ndarray[ndim=1, dtype=np.float32_t] period_dts,
             output="stats"):
        
        # Make sure the number of times and mags matches 
        if len(times) != len(mags):
            return np.zeros([0, 0, 0], dtype=np.float32)
        
        cdef np.ndarray[ndim=1, dtype=np.float32_t] time
        cdef vector[float*] times_ptrs
        cdef vector[size_t] times_lens
        for time_obj in times:
            time = time_obj
            times_ptrs.push_back(&time[0])
            times_lens.push_back(len(time))

        cdef np.ndarray[ndim=1, dtype=np.float32_t] mag
        cdef vector[float*] mags_ptrs
        cdef vector[size_t] mags_lens
        for mag_obj in mags:
            mag = mag_obj
            mags_ptrs.push_back(&mag[0])
            mags_lens.push_back(len(mag))

        # Make sure the individual lengths match
        if any(t != m for t, m in zip(times_lens, mags_lens)):
            return np.zeros([0, 0, 0], dtype=np.float32)

        n_per = len(periods)
        n_pdt = len(period_dts)

        cdef float* ces = self.ce.CalcCEValsBatched(
            times_ptrs, mags_ptrs, times_lens,
            &periods[0], &period_dts[0], n_per, n_pdt,
        )

        cdef np.npy_intp dim[3]
        dim[0] = len(times)
        dim[1] = n_per
        dim[2] = n_pdt

        cdef np.ndarray[ndim=3, dtype=np.float32_t] ces_ndarr = \
            np.PyArray_SimpleNewFromData(3, dim, np.NPY_FLOAT, ces)
        
        if output == 'stats':
            axis = (1, 2)
            means = np.mean(ces_ndarr, axis=axis, dtype=np.float64)
            stds = np.std(ces_ndarr, axis=axis, dtype=np.float64)

            all_stats = []
            for i in range(len(times)):
                argmin = np.unravel_index(
                    np.argmin(ces_ndarr[i]),
                    ces_ndarr[i].shape,
                )

                stats = Statistics(
                    [periods[argmin[0]], period_dts[argmin[1]]],
                    means[i],
                    ces_ndarr[i][argmin],
                    stds[i],
                )

                all_stats.append(stats)
            
            return all_stats
        else:
            raise NotImplementedError('Only "stats" output is implemented')

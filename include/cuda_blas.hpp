#pragma once
#include <cublas_v2.h>
#include <cufft.h>
#include <complex>
#include <type_traits>
#include "cuda_helper.hpp"
static  cublasHandle_t cuBlasHandle() {
    struct wrapper {
        cublasHandle_t handle;
        wrapper() {CUBLAS_CALL(cublasCreate(&handle));}
        ~wrapper() {CUBLAS_CALL(cublasDestroy(handle));}
    };
    static wrapper w;
    return w.handle;
}

template<class T> struct cuda_mapping{using type = T;};
template<> struct cuda_mapping<std::complex<float>>{using type = cuFloatComplex;};
template<> struct cuda_mapping<std::complex<double>>{using type = cuDoubleComplex;};
template<class T> using cuda_t = typename cuda_mapping<T>::type;

#define REPEAT_CODE(T, s,d, z,c, ...) \
    if constexpr(is_s<T>){s(__VA_ARGS__);} \
    else if constexpr(is_d<T>){d(__VA_ARGS__);} \
    else if constexpr(is_c<T> || std::is_same_v<cuFloatComplex, T>){z(__VA_ARGS__);} \
    else if constexpr(is_z<T> || std::is_same_v<cuDoubleComplex, T>){c(__VA_ARGS__);}\
    else{unreachable_constexpr_if();}

namespace cuda
{
    // struct 

    template <typename T> inline void VtAddV1(const int n, const T *x, T *y)
    {
        using TCuda = cuda_t<T>;
        TCuda scalar {1.0};
        REPEAT_CODE(T, cublasSaxpy, cublasDaxpy, cublasCaxpy, cublasZaxpy, cuBlasHandle(), n, &scalar, (const TCuda*)x, 1, (TCuda*)y, 1);
    }
    template <typename T> inline void VtSubV1(const int n, const T *x, T *y)
    {
        using TCuda = cuda_t<T>;
        TCuda scalar {-1.0};
        REPEAT_CODE(T, cublasSaxpy, cublasDaxpy, cublasCaxpy, cublasZaxpy, cuBlasHandle(), n, &scalar, (const TCuda*)x, 1, (TCuda*)y, 1);
    }
}
#include "cuda_basic_operator.h"
#include "cuda_vector.hpp"
#include <cuda_operator.cuh>
#include <span>
//== TODO list:
// 1. expose threadsPerBlock to caller

template<class T> __global__ void vectorAdd(int n, const T* A, const T* B, T* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

template<class T> __global__ void vectorSub(int n, const T* A, const T* B, T* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] - B[i];
}
template<class T> __global__ void vectorMul(int n, const T* A, const T* B, T* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * B[i];
}
template<class T> __global__ void vectorDiv(int n, const T* A, const T* B, T* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] / B[i];
}
template <typename T> void VtAddImpl(const int n, const T *px, T *py)
{
    using TCuda = cuda_t<T>;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cuda::device_vector<T> x, y; 
    T* p = const_cast<T*>(px);
    x << std::span<T>(p, p + n);
    std::span<T> array_y(py, py+n);
    y << array_y;
    vectorAdd<TCuda><<<blocksPerGrid, threadsPerBlock>>>(n, (const TCuda*)y.data(), (const TCuda*)x.data(), (TCuda*)y.data());
    array_y << y;
    CUDA_RT_LAST_ERROR();
}
template <typename T> void VtSubImpl(const int n, const T *x, T *y)
{
    using TCuda = cuda_t<T>;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorSub<TCuda><<<blocksPerGrid, threadsPerBlock>>>(n,  (const TCuda*)y, (const TCuda*)x, (TCuda*)y);
    cudaDeviceSynchronize();
    CUDA_RT_LAST_ERROR();
}
template <typename T> void VtMulImpl(const int n, const T *x, T *y)
{
    using TCuda = cuda_t<T>;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorMul<TCuda><<<blocksPerGrid, threadsPerBlock>>>(n, (const TCuda*)y, (const TCuda*)x, (TCuda*)y);
    cudaDeviceSynchronize();
    CUDA_RT_LAST_ERROR();
}
template <typename T> void VtDivImpl(const int n, const T *x, T *y)
{
    using TCuda = cuda_t<T>;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorDiv<TCuda><<<blocksPerGrid, threadsPerBlock>>>(n, (const TCuda*)y, (const TCuda*)x, (TCuda*)y);
    cudaDeviceSynchronize();
    CUDA_RT_LAST_ERROR();
}
namespace cuda
{
    template<>void VtAdd<float>(const int n, const float *x, float *y){ VtAddImpl(n, x, y);}
    template<>void VtAdd<double>(const int n, const double *x, double *y){ VtAddImpl(n, x, y);}
    template<>void VtAdd<std::complex<float>>(const int n, const std::complex<float> *x, std::complex<float> *y){ VtAddImpl(n, x, y);}
    template<>void VtAdd<std::complex<double>>(const int n, const std::complex<double> *x, std::complex<double> *y){ VtAddImpl(n, x, y);}

    template<>void VtSub<float>(const int n, const float *x, float *y){ VtSubImpl(n, x, y);}
    template<>void VtSub<double>(const int n, const double *x, double *y){ VtSubImpl(n, x, y);}
    template<>void VtSub<std::complex<float>>(const int n, const std::complex<float> *x, std::complex<float> *y){ VtSubImpl(n, x, y);}
    template<>void VtSub<std::complex<double>>(const int n, const std::complex<double> *x, std::complex<double> *y){ VtSubImpl(n, x, y);}

    template<>void VtMul<float>(const int n, const float *x, float *y){ VtMulImpl(n, x, y);}
    template<>void VtMul<double>(const int n, const double *x, double *y){ VtMulImpl(n, x, y);}
    template<>void VtMul<std::complex<float>>(const int n, const std::complex<float> *x, std::complex<float> *y){ VtMulImpl(n, x, y);}
    template<>void VtMul<std::complex<double>>(const int n, const std::complex<double> *x, std::complex<double> *y){ VtMulImpl(n, x, y);}
    
    template<>void VtDiv<float>(const int n, const float *x, float *y){ VtDivImpl(n, x, y);}
    template<>void VtDiv<double>(const int n, const double *x, double *y){ VtDivImpl(n, x, y);}
    template<>void VtDiv<std::complex<float>>(const int n, const std::complex<float> *x, std::complex<float> *y){ VtDivImpl(n, x, y);}
    template<>void VtDiv<std::complex<double>>(const int n, const std::complex<double> *x, std::complex<double> *y){ VtDivImpl(n, x, y);}
}
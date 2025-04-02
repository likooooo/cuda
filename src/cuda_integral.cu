#include <span>
#include <cuda_operator.cuh>
#include "cuda_vector.hpp"
#include "cuda_integral.h"


template <typename T>
__global__ void integral_y_kernel(int2 shape, T* image) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t xsize = shape.x;
    const size_t ysize = shape.y;
    
    if (x < xsize) {
        T* a = image + x;
        for(size_t y = 0; y < ysize - 1; y++) {
            T* b = a + xsize;
            b[0] = b[0] + a[0];
            a = b;
        }
    }
}

template <typename T>
void integral_y_impl(vec2<size_t> shape, T* image) {
    const size_t xsize = shape[1];
    const size_t blockSize = 256;
    const size_t gridSize = (xsize + blockSize - 1) / blockSize;

    cuda::device_vector<T> x;
    std::span<T> buf(image, image + shape[0] * shape[1]);
    x << buf;
    using TCuda = cuda_t<T>;
    int2 s{.x = int(shape[1]), .y=int(shape[0])};
    integral_y_kernel<TCuda><<<gridSize, blockSize>>>(s, (TCuda*)x.data());
    buf << x;
}

template <typename T>
__global__ void integral_x_kernel(int2 shape, T* image) {
    const size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t xsize = shape.x;
    const size_t ysize = shape.y;
    if (y < ysize) {
        T* row = image + y * xsize;
        for(size_t x = 1; x < 2; x++) {
            row[x] = row[x] + row[x - 1];
        }
    }
}

template <typename T>
void integral_x_impl(vec2<size_t> shape, T* image) {
    const size_t ysize = shape[0];
    const size_t blockSize = 256;
    const size_t gridSize = (ysize + blockSize - 1) / blockSize;
    
    cuda::device_vector<T> x;
    std::span<T> buf(image, image + shape[0] * shape[1]);
    x << buf;
    using TCuda = cuda_t<T>;
    cudaDeviceSynchronize();
    int2 s{.x = int(shape[1]), .y=int(shape[0])};
    integral_x_kernel<TCuda><<<gridSize, blockSize>>>(s, (TCuda*)x.data());
    cudaDeviceSynchronize();
    buf << x;
}
namespace cuda
{
    template<> void integral_x<float>(vec2<size_t> shape, float* image){integral_x_impl<float>(shape, image);}
    template<> void integral_x<double>(vec2<size_t> shape, double* image){integral_x_impl<double>(shape, image);}
    template<> void integral_x<complex_t<float>>(vec2<size_t> shape, complex_t<float>* image){integral_x_impl<complex_t<float>>(shape, image);}
    template<> void integral_x<complex_t<double>>(vec2<size_t> shape, complex_t<double>* image){integral_x_impl<complex_t<double>>(shape, image);}

    
    template<> void integral_y<float>(vec2<size_t> shape, float* image){integral_y_impl<float>(shape, image);}
    template<> void integral_y<double>(vec2<size_t> shape, double* image){integral_y_impl<double>(shape, image);}
    template<> void integral_y<complex_t<float>>(vec2<size_t> shape, complex_t<float>* image){integral_y_impl<complex_t<float>>(shape, image);}
    template<> void integral_y<complex_t<double>>(vec2<size_t> shape, complex_t<double>* image){integral_y_impl<complex_t<double>>(shape, image);}
}
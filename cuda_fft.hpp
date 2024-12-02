#pragma once
#include "cuda_basic_operator.h"
#include "cuda_blas.hpp"
#include "cuda_vector.hpp"
#include <cufftXt.h>
#include <cufft.h>
#include <numeric>
#include <memory>
#include <algorithm>

namespace cuda
{
    template<class TSpatial>
    struct fft_io_type{
        using spatial_type = cuda_t<TSpatial>;
        using fourier_type =  cuda_t<complex_t<TSpatial>>;
    };

    template<class T> struct cuFFT;
    template<> struct cuFFT<float> : public fft_io_type<float>
    {
        constexpr static cufftType forward = CUFFT_R2C; 
        constexpr static cufftType backward = CUFFT_C2R; 
        static void exec_forward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecR2C(plan, (spatial_type*)in, (fourier_type*)out));
        }
        static void exec_backward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecC2R(plan, (fourier_type*)in, (spatial_type*)out));
        }
    }; 
    template<> struct cuFFT<double> : public fft_io_type<double>
    {
        constexpr static cufftType forward = CUFFT_D2Z; 
        constexpr static cufftType backward = CUFFT_Z2D; 

        static void exec_forward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecD2Z(plan, (spatial_type*)in, (fourier_type*)out));
        }
        static void exec_backward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecZ2D(plan, (fourier_type*)in, (spatial_type*)out));
        }
    }; 
    template<> struct cuFFT<std::complex<float>> : public fft_io_type<std::complex<float>>
    {
        constexpr static cufftType forward = CUFFT_C2C; 
        constexpr static cufftType backward = CUFFT_C2C; 

        static void exec_forward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecC2C(plan, (spatial_type*)in, (fourier_type*)out, CUFFT_FORWARD));
        }
        static void exec_backward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecC2C(plan, (fourier_type*)in, (spatial_type*)out, CUFFT_INVERSE));
        }
    };
    template<> struct cuFFT<std::complex<double>>  : public fft_io_type<std::complex<double>>
    {
        constexpr static cufftType forward = CUFFT_Z2Z; 
        constexpr static cufftType backward = CUFFT_Z2Z;

        static void exec_forward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecZ2Z(plan, (spatial_type*)in, (fourier_type*)out, CUFFT_FORWARD));
        }
        static void exec_backward(cufftHandle plan, void* in, void* out = nullptr){
            if(nullptr == out) out = in;
            CUFFT_CALL(cufftExecZ2Z(plan, (fourier_type*)in, (spatial_type*)out, CUFFT_INVERSE));
        }
    };



    //== column major
    template<class TAlloc> inline std::vector<typename TAlloc::value_type, TAlloc> make_inplace_fft_vec_with_padding(const std::vector<int>& col_major_dim){
        using T = typename TAlloc::value_type;
        using vec = std::vector<T, TAlloc>;
        auto prod = std::accumulate(col_major_dim.begin() + 1, col_major_dim.end(), (size_t)1, [](auto a, auto b) {return a * b; });
        auto withpadding = (std::is_floating_point_v<T> ? (col_major_dim.front() / 2 + 1) * 2 : col_major_dim.front());
        vec v;  cuda::resize(v, prod * withpadding);
        return v;
    }

    struct cufft_plan_deleter {
        void operator()(cufftHandle* plan) const {
            cufftDestroy(*plan);
            delete plan;
        }
    };
    inline  std::unique_ptr<cufftHandle, cufft_plan_deleter> make_row_major_plan(const std::vector<int>& row_major_dims, cufftType toward /* cuFFT<T>::forward */, int batch_size = 1){
        const int n = static_cast<int>(CUFFT_R2C == toward);
        int prod = std::accumulate(row_major_dims.begin(), row_major_dims.end() - 1, 1, [](int a, int b){return a * b;}) * ( n == 1 ? ((row_major_dims.back() / 2 + 1) * 2) : row_major_dims.back());
        std::unique_ptr<cufftHandle, cufft_plan_deleter> pPlan(new cufftHandle, cufft_plan_deleter());
        CUFFT_CALL(cufftPlanMany(pPlan.get(), row_major_dims.size(), const_cast<int*>(row_major_dims.data()), nullptr, 1, prod, nullptr, 1, prod, toward, batch_size));
        return pPlan;
    }
    inline std::unique_ptr<cufftHandle, cufft_plan_deleter> make_plan(std::vector<int> col_maojr_dims, cufftType toward, int batch_size = 1){
        std::reverse(col_maojr_dims.begin(), col_maojr_dims.end());
        return make_row_major_plan(col_maojr_dims, toward, batch_size);
    }
    
}
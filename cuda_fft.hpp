#pragma once
#include "cuda_basic_operator.h"
#include "cuda_blas.hpp"
#include "cuda_vector.hpp"
#include <cufftXt.h>
#include <cufft.h>
#include <numeric>
#include <algorithm>
#include <memory>

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


    template<class T> inline std::pair<int ,int> cal_layout(const std::vector<int>&  col_major_dim){
        auto prod = std::accumulate(col_major_dim.begin() + 1, col_major_dim.end(), (size_t)1, [](auto a, auto b) {return a * b; });
        auto change_fastest_axis = (std::is_floating_point_v<T> ? (col_major_dim.front() / 2 + 1) * 2 : col_major_dim.front());
        return {change_fastest_axis, prod};
    }

    //== column major
    template<class TAlloc> inline std::vector<typename TAlloc::value_type, TAlloc> make_inplace_fft_vec_with_padding(const std::vector<int>& col_major_dim){
        using T = typename TAlloc::value_type;
        using vec = std::vector<T, TAlloc>;
        auto [withpadding, prod] = cal_layout<T>(col_major_dim);
        vec v;  cuda::resize(v, prod * withpadding);
        return v;
    }
    template<class V1, class V2> inline bool copy_vec_with_padding(V1& output, const V2& input, const std::vector<int>& col_major_dim)
    {
        using T = typename V2::value_type;
        auto [x, y] = cal_layout<complex_t<T>>(col_major_dim);
        int ostride = output.size() * sizeof(typename V1::value_type)/ (sizeof(T) * y);
        if(x > ostride) return false;
        const char* psrc = (const char*)input.data();
        char* pdest = (char*)output.data();
        
        for(int r = 0; r < y; r++){
            std::memcpy(pdest, psrc, x * sizeof(T));
            psrc += x * sizeof(T);
            pdest += ostride *sizeof(T);
        }
        return true;
    }
    struct cufft_plan_deleter {
        void operator()(cufftHandle* plan) const {
            cufftDestroy(*plan);
            delete plan;
        }
    };
    template<class V1, class V2>inline V1 copy_vec_with_padding(const V2& image, const std::vector<int>& col_major_dims)
    {
        V1 image_with_padding = cuda::make_inplace_fft_vec_with_padding<typename V1::allocator_type>(col_major_dims);
        if(image_with_padding.size() == image.size()){
            std::copy(image.begin(), image.end(), image_with_padding.begin());
        }else{
            auto [x, y] = cuda::cal_layout<complex_t<float>>(col_major_dims);
            crop_image(image_with_padding.data(), (col_major_dims.at(0)/2 + 1) * 2, image.data(), x, x, y);
        }
        return image_with_padding;
    }
    template<class V1, class V2>inline V1 copy_vec_from_padding(const V2& image_with_padding, const std::vector<int>& col_major_dims)
    {
        auto [x, y] = cuda::cal_layout<complex_t<float>>(col_major_dims);
        V1 image(x * y);
        if(image_with_padding.size() == image.size()){
            std::copy(image_with_padding.begin(), image_with_padding.end(), image.begin());
        }else{
            crop_image(image.data(), x, image_with_padding.data(), (col_major_dims.at(0)/2 + 1) * 2, x, y);
        }
        return image;
    }
    inline  std::unique_ptr<cufftHandle, cufft_plan_deleter> make_row_major_plan(const std::vector<int>& row_major_dims, cufftType toward /* cuFFT<T>::forward */, int batch_size = 1){
        std::unique_ptr<cufftHandle, cufft_plan_deleter> pPlan(new cufftHandle, cufft_plan_deleter());
        CUFFT_CALL(cufftPlanMany(
            pPlan.get(), row_major_dims.size(), 
            const_cast<int*>(row_major_dims.data()), 
            nullptr, 1, 0, 
            nullptr, 1, 0, 
            toward, batch_size)
        );
        return pPlan;
    }
    inline std::unique_ptr<cufftHandle, cufft_plan_deleter> make_plan(std::vector<int> col_maojr_dims, cufftType toward, int batch_size = 1){
        std::reverse(col_maojr_dims.begin(), col_maojr_dims.end());
        return make_row_major_plan(col_maojr_dims, toward, batch_size);
    }
    
}
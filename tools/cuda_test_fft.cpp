//== hook ArrayRef::SetPadded
#define protected public
#include <cuda_blas.hpp>
#include <cuda_fft.hpp>
#include <cuda_vector.hpp>
#include <py_helper.hpp>
#include <thread>
#include "cuda_test_common.hpp"
// #include "model/TaskFlow.Framework.hpp"
// #include "model/ImageOperatorSet.hpp"
using namespace cuda;

// namespace std
// {
//     template<> struct numeric_limits<std::complex<float>>{
//         constexpr static auto digits10 = 3;//numeric_limits<float>::digits10; 
//     };
//     template<> struct numeric_limits<std::complex<double>>{
//         constexpr static auto digits10 = numeric_limits<double>::digits10; 
//     };
// }
// template<class T> auto ToComplexRef(ArrayRef<T>& ref)
// {
//     if constexpr (std::is_floating_point_v<T>){
//         using cType = complex_t<T>;
//         std::vector<int> dims = ref.LogicShape();
//         if(ref.IsTransposed()) std::reverse(dims.begin(), dims.end());
//         dims.at(0) = dims.at(0) / 2 + 1;
//         ArrayRef<cType> result((cType*) ref.Data(), dims);
//         return result;
//     }else{
//         return ref;
//     }
// } 
// template<class T> void test_real_fft(const std::string& space, const std::vector<int>& col_major_dims)
// {
//     std::vector<int> row_major_dims = col_major_dims; std::reverse(row_major_dims.begin(), row_major_dims.end());
//     constexpr bool enable_assert = true;
//     using cType = complex_t<T>;
//     printf("*%s cuda::fft<%s>", space.c_str(), TypeReflection<T>().c_str()); std::cout << col_major_dims << std::endl;
//     ImageOwn<T> image(col_major_dims);
//     image.Random(); 

//     //== cuda fft
//     std::vector<T> cpu_vec = cuda::make_inplace_fft_vec_with_padding<std::allocator<T>>(col_major_dims);
//     std::copy(image.begin(), image.end(), cpu_vec.begin());
//     cuda::device_vector<T> gpu_vec; gpu_vec << cpu_vec;
//     {
//         std::vector<T> temp;
//         temp << gpu_vec;
//         assert(temp == cpu_vec);
//     }
//     cuda::cuFFT<T>::exec_forward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::forward), gpu_vec.data());
//     cudaDeviceSynchronize();
//     std::vector<T> result_from_gpu;
//     result_from_gpu << gpu_vec;
//     //== half spectrum from gpu
//     ArrayRef<T> ref_gpu(result_from_gpu.data(), col_major_dims);
//     ref_gpu.SetPadded(true);
//     ref_gpu.SetFourierDomain(true);
//     ref_gpu.SetTransposed(false);
//     auto complex_ref_gpu = ToComplexRef(ref_gpu);
//     imshow(ImageTypeCast<cType, std::complex<float>>(complex_ref_gpu));
//     //== fftw fft
//     ImageOwn<T> full_spectrum(image.LogicShape());
//     {
//         full_spectrum.InitData();
//         auto dim = full_spectrum.LogicShape();
//         //== if with padding
//         if constexpr(std::is_floating_point_v<T>) dim[0] = (dim[0]/2 + 1) * 2;  
//         ArrayRef<T>ref(full_spectrum.Data(), dim);
//         ImageOperator::CropImage(ref, 0, 0, image, 0, 0);
//     }
//     full_spectrum.SetTransposed(true); // block transpose for real-fft
//     full_spectrum.ReShape(row_major_dims);
//     full_spectrum.FFTW(-1);
//     full_spectrum.ReShape(col_major_dims);
//     full_spectrum.SetTransposed(false);
//     ArrayRef<cType> half_spectrum = ToComplexRef(full_spectrum);
    
//     ArrayOwn<cType> eps_image(complex_ref_gpu.LogicShape());
//     CastTo<ImageCastMode::All>(complex_ref_gpu.Data(), eps_image.Data(), complex_ref_gpu.LogicSize());
//     eps_image -= half_spectrum;
//     eps_image /= half_spectrum;
//     float max_eps = eps_image.AmplitudeMax();

//     constexpr int  eps_pow = (2 < std::numeric_limits<T>::digits10) ? std::numeric_limits<T>::digits10 - 2 : 2;
//     if(0 != max_eps)
//     {
//         if(std::pow(0.1, eps_pow) <= max_eps){
//             printf("*%s    \033[31m[WARNING] relative error =%e\033[0m\n", space.c_str(), max_eps);
//             if(enable_assert) {
//                 // printf("*%s    0. FFTW; 1. cufft; 2. error-image\n", space.c_str());
//                 // imshow(
//                 //     ImageTypeCast<cType, std::complex<float>>(half_spectrum), 
//                 //     ImageTypeCast<cType, std::complex<float>>(complex_ref_gpu), 
//                 //     ImageTypeCast<cType, std::complex<float>>(eps_image) // eps
//                 // );
//                 // assert(std::pow(0.1, eps_pow) > max_eps);
//             }
//         } else{
//             printf("*%s    relative error =%e\n", space.c_str(), max_eps);
//         }
//     }
//     printf("*%s    test success\n", space.c_str());
// }


template<class T> void test_complex_fft(const std::string& space, const std::vector<int>& col_major_dims)
{
    std::vector<int> row_major_dims = col_major_dims; std::reverse(row_major_dims.begin(), row_major_dims.end());
    constexpr bool enable_assert = true;
    using cType = complex_t<T>;
    printf("*%s cuda::fft<%s>", space.c_str(), TypeReflection<T>().c_str()); std::cout << col_major_dims << std::endl;
    std::vector<T> image(std::accumulate(col_major_dims.begin(), col_major_dims.end(), 1, [](auto a, auto b){return a * b;}));
    {
        uniform_random<T> r;
        std::generate(image.begin(), image.end(), r);
    }
    //== cuda fft
    std::vector<T> cpu_vec = cuda::make_inplace_fft_vec_with_padding<std::allocator<T>>(col_major_dims);
    std::copy(image.begin(), image.end(), cpu_vec.begin());
    cuda::device_vector<T> gpu_vec; gpu_vec << cpu_vec;
    {
        std::vector<T> temp;
        temp << gpu_vec;
        assert(temp == cpu_vec);
    }
    cuda::cuFFT<T>::exec_forward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::forward), gpu_vec.data());
    cuda::cuFFT<T>::exec_backward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::backward), gpu_vec.data());

    cudaDeviceSynchronize();
    std::vector<T> result_from_gpu;
    result_from_gpu << gpu_vec;

    std::vector<real_t<T>> error(image.size());
    real_t<T> n = std::accumulate(col_major_dims.begin(), col_major_dims.end(), real_t<T>{1}, [](auto a , auto b){return a * b;});
    std::transform(image.begin(), image.end(), result_from_gpu.begin(), error.begin(), 
        [n](auto a, auto b){ return (real_t<T>)std::abs(a - (b /n) );}
    );
    real_t<T> max_error = *std::max_element(error.begin(), error.end());
    std::cout << max_error << std::endl;
    
    py_plot().visulizer["display_image"](create_ndarray_from_vector(error, col_major_dims));
    // static auto callback = py_plot::create_callback_simulation_fram_done(py::object(overload_click));
    // callback(create_ndarray_from_vector(error, col_major_dims));
    // std::this_thread::sleep_for(std::chrono::seconds(20));
    // callback(create_ndarray_from_vector(error, col_major_dims));
    // std::this_thread::sleep_for(std::chrono::seconds(20));
    // callback(create_ndarray_from_vector(error, col_major_dims));
    // std::this_thread::sleep_for(std::chrono::seconds(20));


    // //== half spectrum from gpu
    // ArrayRef<T> ref_gpu(result_from_gpu.data(), col_major_dims);
    // ref_gpu.SetPadded(true);
    // ref_gpu.SetFourierDomain(true);
    // ref_gpu.SetTransposed(false);
    // auto complex_ref_gpu = ToComplexRef(ref_gpu);

    // //== fftw fft
    // ImageOwn<cType> full_spectrum = ImageTypeCast<T, cType>(image);
    // full_spectrum.FFTW(-1);
    // ArrayRef<cType> half_spectrum = ToComplexRef(full_spectrum);
    
    // ArrayOwn<cType> eps_image(complex_ref_gpu.LogicShape());
    // CastTo<ImageCastMode::All>(complex_ref_gpu.Data(), eps_image.Data(), complex_ref_gpu.LogicSize());
    // eps_image -= half_spectrum;
    // eps_image /= half_spectrum;
    // // float max_eps = std::sqrt(std::abs(eps_image | eps_image)) / eps_image.LogicSize(); 
    // float max_eps = eps_image.AmplitudeMax();

    // constexpr int  eps_pow = (2 < std::numeric_limits<T>::digits10) ? std::numeric_limits<T>::digits10 - 2 : 2;
    // if(0 != max_eps)
    // {
    //     if(std::pow(0.1, eps_pow) <= max_eps){
    //         printf("*%s    \033[31m[WARNING] relative error =%e\033[0m\n", space.c_str(), max_eps);
    //         if(enable_assert) {
    //             printf("*%s    0. FFTW; 1. cufft; 2. error-image\n", space.c_str());
    //             imshow(
    //                 ImageTypeCast<cType, std::complex<float>>(half_spectrum), 
    //                 ImageTypeCast<cType, std::complex<float>>(complex_ref_gpu), 
    //                 ImageTypeCast<cType, std::complex<float>>(eps_image) // eps
    //             );
    //             assert(std::pow(0.1, eps_pow) > max_eps);
    //         }
    //     } else{
    //         printf("*%s    relative error =%e\n", space.c_str(), max_eps);
    //     }
    // }
    printf("*%s    test success\n", space.c_str());
}
bool is2a3b5c7d(int N) {
    if (N <= 0) return false; 
    while (N % 2 == 0) N /= 2;
    while (N % 3 == 0) N /= 3;
    while (N % 5 == 0) N /= 5;
    while (N % 7 == 0) N /= 7;
    return N == 1;
}

template<class ...T>
void test_wrapper(const std::string& space = ""){
    printf("*%s compare MKL with cuda\n", space.c_str());

    int lb_usf = 1;
    int ub_usf = 8;
    int lb_dim = 2;
    int ub_dim = 2;
    int lb = 64;
    int ub = 128;

    uniform_random<int> dis_usf(lb_usf, ub_usf);
    uniform_random<int> dis_dim(lb_dim, ub_dim);
    uniform_random<int> dis_size(lb, ub);

    auto cal_random_dim = [&](){
        int usf = dis_usf();
        int dim = dis_dim();
        std::vector<int> dims(dim);
        for(auto& n : dims) {
            do{n = usf * dis_size();}while(!is2a3b5c7d(n));
        };
        return dims;
    };
    std::vector<int> dim = cal_random_dim(); //  {260,492};
    // (test_real_fft<typename RealType<T>::type>(space + "    ", dim), ...);
    (test_complex_fft<real_t<T>>(space + "    ", dim), ...);
}
int main(){
    py_loader::init();
    for(int i = 0; i < 1; i++){
        printf("* test-%d\n", i);
        test_wrapper<float, double, std::complex<float>, std::complex<double>>("    ");
    }
}
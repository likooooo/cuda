//== hook ArrayRef::SetPadded
#define protected public
#include <cuda_blas.hpp>
#include <cuda_fft.hpp>
#include <cuda_vector.hpp>
#include <thread>
#include "cuda_test_common.hpp"
using namespace cuda;
template <class T>
void rect_image(T *p, int width, int height, int rec_size)
{
    for (int y = (height - rec_size) / 2; y < (height + rec_size) / 2; y++)
    {
        for (int x = (width - rec_size) / 2; x < (width + rec_size) / 2; x++)
        {
            p[width * y + x] = T(1);
        }
    }
}
template<class T> void test_fft_check(const std::string& space, std::vector<int> col_major_dims)
{
    printf("*%s test fft<%s>", space.c_str(), TypeReflection<T>().c_str()); std::cout << col_major_dims << std::endl;
    auto [x, y] = cuda::cal_layout<complex_t<T>>(col_major_dims);
    std::vector<T> image(x * y);
    {
        uniform_random<T> r(1e-2, 1);
        std::generate(image.begin(), image.end(), r);
    }
    auto image_with_padding = cuda::copy_vec_with_padding<std::vector<T>>(image, col_major_dims);
    cuda::device_vector<T> gpu_vec; gpu_vec << image_with_padding;
    cuda::cuFFT<T>::exec_forward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::forward), gpu_vec.data());
    std::vector<complex_t<T>> freq;
    freq << gpu_vec;
    catch_py_error(assert(py_loader(".")["cuda_test_fft_golden"]["check_rfft"](
        create_ndarray_from_vector(image, col_major_dims),
        create_ndarray_from_vector(freq, {static_cast<int>(freq.size())/y, y}))
    ));
    printf("*%s    test success\n", space.c_str());
}
template<class T> void test_fft_ifft(const std::string& space, std::vector<int> col_major_dims)
{
    using cType = complex_t<T>;
    printf("*%s test ifft(fft<%s>(I)) ", space.c_str(), TypeReflection<T>().c_str()); std::cout << col_major_dims << std::endl;
    auto [x, y] = cuda::cal_layout<complex_t<T>>(col_major_dims);
    std::vector<T> image(x * y);
    {
        uniform_random<T> r(1e-2, 1);
        std::generate(image.begin(), image.end(), r);
    }
    //== prepare forward-data
    auto cpu_vec = cuda::copy_vec_with_padding<std::vector<T>>(image, col_major_dims);
    cuda::device_vector<T> gpu_vec; gpu_vec << cpu_vec;
    {
        std::vector<T>temp; temp << gpu_vec;
        assert(temp == cpu_vec);
    }
    //== fft & ifft
    cuda::cuFFT<T>::exec_forward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::forward), gpu_vec.data());
    cudaDeviceSynchronize();
    cuda::cuFFT<T>::exec_backward(*cuda::make_plan(col_major_dims, cuda::cuFFT<T>::backward), gpu_vec.data());
    cudaDeviceSynchronize();
    std::vector<T> result_from_gpu;
    result_from_gpu << gpu_vec;
    // normalization
    for(auto& n : result_from_gpu) n/=static_cast<real_t<T>>(image.size());
    result_from_gpu = copy_vec_from_padding<std::vector<T>>(result_from_gpu, col_major_dims);
    auto golden_checker = py_loader(".")["cuda_test_fft_golden"];
    auto ref_result = create_ndarray_from_vector(result_from_gpu, col_major_dims); 
    auto ref_golden = create_ndarray_from_vector(image, col_major_dims); 
    bool check_pass = false;
    catch_py_error(check_pass = golden_checker["custom_allclose"](ref_result, ref_golden));
    if(!check_pass) catch_py_error(golden_checker["print_result"](ref_golden, ref_result, ref_golden));
    assert(check_pass);
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
    
    (test_fft_check<T>(space, dim), ...);
    (test_fft_ifft<T>(space, dim), ...);
}
int main(){
    py_loader::init();
    printf("* Compare numpy.fft(FFTW) with cuda\n");
    for(int i = 0; i < 10; i++){
        printf("* iter-%d\n", i);
        test_wrapper<float, double, std::complex<float>, std::complex<double>>("    ");
    }
}
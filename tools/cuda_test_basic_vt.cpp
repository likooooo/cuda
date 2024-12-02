#include <cuda_basic_operator.h>
#include <cuda_vector.hpp>
#include <limits> 
#include <cmath>
#include "cuda_test_common.hpp"

template<class T> void test_add(const std::string space = ""){
    printf("*%s cuda::VtAdd<%s>\n", space.c_str(), TypeReflection<T>().c_str());
    auto [array_x, array_y] = init_input_vector<T>();

    cuda::device_vector<T> x, y; x << array_x; y << array_y;
    cuda::VtAdd(x.size(), x.data(), y.data());
    cuda::pageable_vector<T> result_from_gpu; result_from_gpu << y;
    
    std::transform(array_y.begin(), array_y.end(), array_x.begin(), array_y.begin(),[](auto a, auto b){return a + b;});
    assert(array_y == result_from_gpu);
    printf("*%s    test success\n", space.c_str());
}
template<class T> void test_sub(const std::string space = ""){
    printf("*%s cuda::VtSub<%s>\n", space.c_str(), TypeReflection<T>().c_str());
    auto [array_x, array_y] = init_input_vector<T>();

    cuda::device_vector<T> x, y; x << array_x; y << array_y;
    cuda::VtSub(x.size(), x.data(), y.data());
    cuda::pageable_vector<T> result_from_gpu; result_from_gpu << y;
    
    std::transform(array_y.begin(), array_y.end(), array_x.begin(), array_y.begin(),[](auto a, auto b){return a - b;});
    assert(array_y == result_from_gpu);
    printf("*%s    test success\n", space.c_str());
}
template<class T> void test_mul(const std::string space = ""){
    printf("*%s cuda::VtMul<%s>\n", space.c_str(), TypeReflection<T>().c_str());
    auto [array_x, array_y] = init_input_vector<T>();

    cuda::device_vector<T> x, y; x << array_x; y << array_y;
    cuda::VtMul(x.size(), x.data(), y.data());
    cuda::pageable_vector<T> result_from_gpu; result_from_gpu << y;
    
    std::transform(array_y.begin(), array_y.end(), array_x.begin(), array_y.begin(),[](auto a, auto b){return a * b;});
    auto [max_eps, threshold] = compare_result(array_y, result_from_gpu);
    if(0 != max_eps){
        printf("*%s    \033[31m[WARNING] relative error:%e, rounding error:%e\033[0m\n", space.c_str(), max_eps, threshold);
        assert(threshold > max_eps);
    }
    printf("*%s    test success\n", space.c_str());
}
template<class T> void test_div(const std::string space = ""){
    printf("*%s cuda::VtDiv<%s>\n", space.c_str(), TypeReflection<T>().c_str());
    auto [array_x, array_y] = init_input_vector<T>();

    cuda::device_vector<T> x, y; x << array_x; y << array_y;
    cuda::VtDiv(x.size(), x.data(), y.data());
    cuda::pageable_vector<T> result_from_gpu; result_from_gpu << y;
    
    std::transform(array_y.begin(), array_y.end(), array_x.begin(), array_y.begin(),[](auto a, auto b){return a / b;});
    auto [max_eps, threshold] = compare_result(array_y, result_from_gpu);
    if(0 != max_eps){
        printf("*%s    \033[31m[WARNING] relative error:%e, rounding error:%e\033[0m\n", space.c_str(), max_eps, threshold);
        assert(threshold > max_eps);
    }
    printf("*%s    test success\n", space.c_str());
}

template<class ...T>
void test_wrapper(const std::string& space = "")
{
    (test_add<T>(space), ...);
    (test_sub<T>(space), ...);
    (test_mul<T>(space), ...);
    (test_div<T>(space), ...);
}
int main() 
{
    for(int i = 0; i < 10; i++){
        printf("* compare gpu with cpu test-%d (basic calculation)\n", i);
        test_wrapper<float, double, std::complex<float>, std::complex<double>>("    ");
    }
    return 0;
}
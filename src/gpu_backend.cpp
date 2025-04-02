#include <gpu_backend.hpp>
#include <cuda_basic_operator.h>
#include <cuda_vector.hpp>
namespace uca
{
    template<class T> void gpu_backend_impl(gpu_backend<T>& gpu)
    {
        cpu.enable =  true;
        gpu.VtAdd = cuda::VtAdd<T>;
    }
    template<> gpu_backend<float>::gpu_backend(){gpu_backend_impl(*this);}
    template<> gpu_backend<double>::gpu_backend(){gpu_backend_impl(*this);}
    template<> gpu_backend<complex_t<float>>::gpu_backend(){gpu_backend_impl(*this);}
    template<> gpu_backend<complex_t<double>>::gpu_backend(){gpu_backend_impl(*this);}
}
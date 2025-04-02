#include <type_traist_notebook/type_traist.hpp>
#include <type_traist_notebook/uca/backend.hpp>
namespace uca
{  
    template<class T> struct gpu_backend : backend<T>
    {
        using value_type = T;
        using alloc_type = std::allocator<T>;
        gpu_backend();
        static gpu_backend& ref()
        {
            static gpu_backend gpu;
            return gpu;
        }
    };
    template<class T> using gpu = gpu_backend<T>;
}
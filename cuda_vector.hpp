#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <type_traits>
#include <cstring>
// #include "model/Images.h"
#include "cuda_helper.hpp"

// 1D
// pinned -> device   : cudaMemcpyAsync : cudaMemcpyHostToDevice
// device -> pinned   : cudaMemcpyAsync : cudaMemcpyDeviceToHost //cudaDeviceSynchronize
// host_pageable -> device : cudaMemcpy : cudaMemcpyHostToDevice
// device -> host_pageable : cudaMemcpy : cudaMemcpyDeviceToHost 
// device -> device        : cudaMemcpy : cudaMemcpyDeviceToDevice
// host -> host            : std::memcpy // cudaDeviceSynchronize

// 2D
// cudaMallocPitch, cudaMemcpy2D
// cudaMallocArray, cudaArraySetComponent, cudaCreateChannelDesc, cudaGetArrayChannelDesc

//TODO :
// 1. event
// 2. resource

namespace cuda
{
    enum class memory_type {
        device, 
        pinned,
        pageable, 
        managed,
    };
    
    template<typename T, memory_type mem_t = memory_type::managed> class cuda_allocator : public std::allocator<T> {
    public:
        static_assert(std::is_trivially_copyable_v<T>, "T should be trivial-type");
        static_assert(mem_t == memory_type::pageable || std::is_trivially_copyable_v<T>);

        //== std::allocator_traits
        using base_type = std::allocator<T>;
        typedef T* pointer;
        typedef T        value_type;
        typedef size_t     size_type;
        typedef ptrdiff_t  difference_type;
        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;
        template<typename U> struct rebind { using other = cuda_allocator<U, mem_t>;};

        cuda_allocator() = default;
        template<class T1> cuda_allocator(const cuda_allocator<T1>& a):std::allocator<T1>(a){}
        cuda_allocator& operator=(const cuda_allocator&) = default;
        ~cuda_allocator() = default;
        
        pointer allocate(std::size_t n, const void* = nullptr) {
            std::size_t size = n * sizeof(T);
            void* ptr = nullptr;
            if constexpr (mem_t == memory_type::device) {
                CUDA_RT_CALL(cudaMalloc(&ptr, size));
            }
            else if constexpr (mem_t == memory_type::pinned) {
                CUDA_RT_CALL(cudaMallocHost(&ptr, size));
            }
            else if constexpr (mem_t == memory_type::managed) {
                CUDA_RT_CALL(cudaMallocManaged(&ptr, size));
            }
            else if constexpr (mem_t == memory_type::pageable){
                ptr = base_type::allocate(n);
            }
            return static_cast<pointer>(ptr);
        }
        void deallocate(pointer p, std::size_t n) {
            if (p == nullptr) return;
            if constexpr (mem_t == memory_type::device) {
                CUDA_RT_CALL(cudaFree(p));
            }
            else if constexpr (mem_t == memory_type::pinned) {
                CUDA_RT_CALL(cudaFreeHost(p));
            }
            else if constexpr (mem_t == memory_type::managed) {
                CUDA_RT_CALL(cudaFree(p));
            }
            else if constexpr (mem_t == memory_type::pageable){
                base_type::deallocate(p, n);
            }
        }
    };
    template<class T, memory_type N>  using vector = std::vector<T, cuda_allocator<T, N>>;
    template<class T> using device_vector = std::vector<T, cuda_allocator<T, memory_type::device>>;
    template<class T> using pinned_vector = std::vector<T, cuda_allocator<T, memory_type::pinned>>;
    template<class T> using managed_vector = std::vector<T, cuda_allocator<T, memory_type::managed>>;
    template<class T> using pageable_vector = std::vector<T>;
     
    template<class T, memory_type M> constexpr static bool is_target_vec_v = std::is_same_v<
        std::vector<typename T::value_type, cuda_allocator<typename T::value_type, M>>, T> || 
        std::is_same_v<cuda_allocator<typename T::value_type, M>, T>;
    template<class T> constexpr static bool is_device_vec_v = is_target_vec_v<T, memory_type::device>;
    template<class T> constexpr static bool is_pinned_vec_v = is_target_vec_v<T, memory_type::pinned>;

    template<class T, class TAlloc> inline void resize(std::vector<T, TAlloc>& vec, int n) {
        if constexpr (is_device_vec_v<std::vector<T, TAlloc>>) {
            union converter 
            {
                std::array<T*, 3> p;
                std::vector<T, TAlloc> v;
                static_assert(sizeof(std::array<T*, 3>) == sizeof(std::vector<T, TAlloc>));
            };
            //== UB ??
            converter& c = reinterpret_cast<converter&>(vec);
            int old_size = vec.capacity();
            if (n > old_size) vec.reserve(n);
            // printf("(%p, %p, %p), (%p, %p, %p)\n", c.p[0], c.p[1], c.p[2], vec.data(), vec.data() + vec.size(),  vec.data() + vec.capacity());
            CUDA_RT_CALL(cudaMemset(c.p[0], 0, sizeof(T) * n));
            c.p[1] = c.p[0] + n;
        }
        else {
            vec.resize(n);
        }
    }
    template<class T> inline T* _ZeroCopyGPUPtr(T* pCPU){
        T* pGPU{nullptr};
        cudaHostGetDevicePointer((void **)&pGPU, (void *)pCPU, 0);
        return pGPU;
    }
    template<class T> inline T* ZeroCopyGPUPtr(pinned_vector<T>& vec){ return _ZeroCopyGPUPtr(vec.data()); }
};

struct SyncProxy
{
    SyncProxy() = default;
    ~SyncProxy(){
        if(valid) CUDA_RT_CALL(cudaDeviceSynchronize());
    }
    SyncProxy& operator & (SyncProxy&& proxy){
        proxy.valid = false;
        return *this;
    }
    bool valid{true};
    // TODO : proxy operator
};

template<class TAlloc1, class TAlloc2> inline auto operator << (std::vector<typename TAlloc1::value_type, TAlloc1>& dest, const std::vector<typename TAlloc2::value_type, TAlloc2>& src){
    using T = typename TAlloc2::value_type;
    using T2 = typename TAlloc1::value_type;

    cuda::resize(dest, src.size() * sizeof(T)/ sizeof(T2));
    constexpr bool has_device = cuda::is_device_vec_v<TAlloc1> || cuda::is_device_vec_v<TAlloc2>;
    constexpr bool has_pinned = cuda::is_pinned_vec_v<TAlloc1> || cuda::is_pinned_vec_v<TAlloc2>;

    if constexpr(has_device){
        if constexpr(has_pinned){
            CUDA_RT_CALL(cudaMemcpyAsync(dest.data(), src.data(), sizeof(T) * src.size(), cudaMemcpyDefault, 0));
        }else{
            CUDA_RT_CALL(cudaMemcpy(dest.data(), src.data(), sizeof(T) * src.size(), cudaMemcpyDefault));
        }
        return SyncProxy();
    } 
    else {
        CUDA_RT_CALL(cudaMemcpy(dest.data(), src.data(), sizeof(T) * src.size(), cudaMemcpyDefault));
    }
}

template<class T>inline void crop_image(T* pOut, const int ostride, const T* pIn, const int istride, const int sizex, const int sizey)
{
    //  cblas_scopy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
    for(int y = 0; y < sizey; y++, pOut += ostride, pIn += istride){
        std::memcpy(pOut, pIn, sizex * sizeof(T));
    }
}

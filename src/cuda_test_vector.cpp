#include <cuda_vector.hpp>
#include <assert.h>
#include <algorithm>
#include <numeric>
#include <complex>
#include "cuda_test_common.hpp"

using namespace cuda;
void test_allocation_and_deallocation() {
    auto test = [](auto vec) { cuda::resize(vec, 1024); vec.clear(); };
    test(pinned_vector<int>());
    test(managed_vector<int>());
    test(device_vector<int>()); 
    std::cout << "test allocation and deallocation success" << std::endl;
}

void test_memory_transfer() {
    // copy pageable-vector to device
    pageable_vector<int> data(1024); std::iota(data.begin(), data.end(), 0);
    device_vector<int> cuda_vec_device;
    cuda_vec_device << data;
    {
        pageable_vector<int> temp;
        temp << cuda_vec_device;
        assert(temp == data);
    }

    // pinned memory (RAM mapping to GPU ) test
    pinned_vector<std::complex<double>> cuda_vec_host; cuda::resize(cuda_vec_host, 1024);
    cuda_vec_host[0] = std::complex<double>(-1); 
    cuda_vec_host.at(1023) = std::complex<double>(1);
    {
        pageable_vector<std::complex<double>> temp;
        temp << cuda_vec_host;
        assert(temp.at(0) == std::complex<double>(-1));
        assert(temp.at(1023) == std::complex<double>(1));
    }
    // managed memory (RAM will be migrated to GPU)
    managed_vector<float> cuda_vec_managed; cuda::resize(cuda_vec_managed, 1024);
    cuda_vec_managed[0] = -1; 
    cuda_vec_managed.at(1023) = 1;
    {
        pageable_vector<float> temp;
        temp << cuda_vec_managed;
        std::vector<float> diff;
        std::set_difference(temp.begin(), temp.end(), cuda_vec_managed.begin(), cuda_vec_managed.end(), std::back_inserter(diff));
        assert(diff.size() == 0);
        assert(temp.at(0) == -1);
        assert(temp.at(1023) == 1);
    }
    std::cout << "test memory transfer success" << std::endl;
}

struct cu_event{
    cudaEvent_t e{nullptr};
    cu_event() {CUDA_RT_CALL(cudaEventCreate(&e));}  
    cu_event(cu_event&& temp){std::swap(e, temp.e);}
    ~cu_event(){if(e)CUDA_RT_CALL(cudaEventDestroy(e));}
    void record(cudaStream_t stream = nullptr){CUDA_RT_CALL(cudaEventRecord(e, stream));}

    float operator-(const cu_event& event){
        float elapsedTimeInMs;   
        CUDA_RT_CALL(cudaEventElapsedTime(&elapsedTimeInMs, e, event.e));
        return elapsedTimeInMs;
    }
};
constexpr unsigned MEMCOPY_ITERATIONS = 100;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wliteral-suffix"
constexpr long long operator"" KB(unsigned long long value) {
    return value * 1024;
}
constexpr long long operator"" MB(unsigned long long value) {
    return value * 1024 * 1024;
}
constexpr long long operator"" GB(unsigned long long value) {
    return value * 1024 * 1024 * 1024;
}
#pragma warning(pop)
template<cuda::memory_type from, cuda::memory_type to> float test_transfer_bandwidth_impl(unsigned int memSize, const std::string& space) {
    int element_count = memSize / sizeof(int);
    cuda::vector<int, from> in; cuda::resize(in, element_count);
    cuda::vector<int, to> out;cuda::resize(out, element_count);
    std::vector<int> golden_data(element_count); std::iota(golden_data.begin(), golden_data.end(), 0);
    in << golden_data;
    constexpr std::array<const char*, 4> str{"device", "pinned", "pageable", "managed"};
    printf("*%s    from %s to %s\n", space.c_str(), str.at((int)from), str.at((int)to));
    float elapsedTimeInMs = 0;
    if constexpr(from == cuda::memory_type::device || to == cuda::memory_type::device){
        cu_event start, stop;
        {
            start.record();
            SyncProxy sync;
            for(int i = 0; i < MEMCOPY_ITERATIONS; i++) sync & (out << in);
            stop.record();
        }
        elapsedTimeInMs += (start - stop);
    }else{
        uint64_t t = get_microseconds();
        for(int i = 0; i < MEMCOPY_ITERATIONS; i++){
            out << in;
        }
        elapsedTimeInMs += (get_microseconds() - t) * 1e-3f;
    }
    float bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (float)1e9;
    bandwidthInGBs = bandwidthInGBs / (elapsedTimeInMs * 1e-3);
    printf("*%s         Transfer Size = %u (MB), Bandwidth= %f (GB/s)\n", space.c_str(), memSize * MEMCOPY_ITERATIONS /1024/1024, bandwidthInGBs);
    std::vector<int> temp; temp << out;
    assert(temp == golden_data);
    return bandwidthInGBs;
}

template<class T> struct combination_generator
{
    template <T ...N, size_t ...Is>
    constexpr static auto generate_value_pairs(std::index_sequence<Is...>) {
        constexpr T values[] = { N... };
        return std::array<std::pair<T, T>, sizeof...(N) * sizeof...(N)>{
            { std::pair<T, T>{values[Is / sizeof...(N)], values[Is % sizeof...(N)]}... }
        };
    }
    template<T ...N>
    constexpr static auto get_combinations() {
        return generate_value_pairs<N...>(std::make_index_sequence<sizeof...(N) * sizeof...(N)>{});
    }
    template<T ...N, size_t ...Is>
    constexpr static void test_transfer_bandwidth_for_all_combinations(unsigned int memSize, const std::string& space, std::index_sequence<Is...>) {
        constexpr auto pairs = get_combinations<N...>();
        (test_transfer_bandwidth_impl<pairs[Is].first, pairs[Is].second>(memSize, space), ...);
    }
};
void test_bandwidth(){
    printf("* bandwidth test\n");
    combination_generator<cuda::memory_type>::test_transfer_bandwidth_for_all_combinations<
        cuda::memory_type::device,
        cuda::memory_type::pinned,
        cuda::memory_type::pageable,
        cuda::memory_type::managed
    >(32MB, "    ",std::make_index_sequence<4 * 4>{});
}

int main() {
    test_allocation_and_deallocation();
    test_memory_transfer();
    test_bandwidth();
    return 0;
}
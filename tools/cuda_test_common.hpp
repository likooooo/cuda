#pragma once
#include <random>
#include <complex>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <memory>
#include <fstream>
#include <type_traist_notebook.hpp>

template <typename T, typename = void>
struct is_distribution : std::false_type {};

template <typename T>
struct is_distribution<T, std::void_t<typename T::param_type>> : std::true_type {};
template<class T, class = void> struct uniform_distribution;
template<class T> struct uniform_distribution<T, std::enable_if_t<std::is_integral_v<T>>>{
    using type = std::uniform_int_distribution<T>;
};   
template<class T> struct uniform_distribution<T, std::enable_if_t<std::is_floating_point_v<T>>>{
    using type = std::uniform_real_distribution<T>;
};

template <class T, class TDistribution, class TEngine = std::mt19937>
struct random_callable {
    static_assert(is_distribution<TDistribution>::value, "TDistribution must be a distribution type");
    using result_type = typename TDistribution::result_type;

    random_callable(result_type lb = result_type{0}, result_type ub = result_type{1}) : engine_(std::random_device{}()), distribution_(TDistribution(lb, ub)){}
    T operator()(){
        if constexpr(is_complex_v<T>){
            return T{distribution_(engine_), distribution_(engine_)};
        }else{
            return distribution_(engine_);
        }
    }
private:
    TEngine engine_;
    TDistribution distribution_;
};

template<class T> using uniform_random = random_callable<T, typename uniform_distribution<real_t<T>>::type, std::mt19937>;

template<class V1, class V2> inline auto compare_result(const V1& a, const V2& b){
    using real = real_t<typename V1::value_type>;
    const real threshold = std::pow(real(0.1), std::numeric_limits<real>::digits10);
    if(a == b) return std::pair<real, real>{real(0), threshold};
    std::vector<real> eps(a.size());
    std::transform(a.begin(), a.end(), b.begin(), eps.begin(),[](const auto a, const auto b){return std::abs((a - b) / a);});
    const real max_eps = *std::max_element(eps.begin(), eps.end());
    return std::pair<real, real>{max_eps, threshold};
}

template<class T> inline auto init_input_vector(int size = 3 * 4){
    cuda::pageable_vector<T> array_x(size);
    cuda::pageable_vector<T> array_y(size);
    uniform_random<T> r;
    std::generate(array_x.begin(), array_x.end(), r);
    std::generate(array_y.begin(), array_y.end(), r);
    return std::make_tuple(array_x, array_y);
}

inline uint64_t get_microseconds(){
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

struct Image{
    template<class T> constexpr static unsigned make_flag()
    {
        union{
            struct {
                uint8_t size; uint8_t status; // <is_intger, is_floatting_point, is_complex>
            }flag_meta;
            unsigned n;
        }u;
        u.flag_meta.size = sizeof(T);
        u.flag_meta.status = (uint8_t)std::is_integral_v<T> + 
            (uint8_t)std::is_floating_point_v<T> * 2 +
            (uint8_t)std::is_complex_v<T> * 4;
        return u.n;
    }
    unsigned xsize{0}, ysize{0};
    unsigned flag{0};
    char ptr[0];
    unsigned bytes_of_bitmap()const{ return xsize * ysize * (flag & 0xff);}
    void serialize_image(std::ostream& ofs)const{
        const Image& img = *this;
        ofs.write(reinterpret_cast<const char*>(&img), sizeof(Image) + img.bytes_of_bitmap());
    }
    static std::unique_ptr<Image> deserialize(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error( (std::ostringstream() << "Failed to open file for reading." << filename).str());
        Image img_header;
        ifs.read(reinterpret_cast<char*>(&img_header), sizeof(Image));
        std::unique_ptr<Image> pImage((Image*)std::malloc(sizeof(Image) + img_header.bytes_of_bitmap()));
        *pImage = img_header;
        ifs.read(reinterpret_cast<char*>(pImage->ptr), img_header.bytes_of_bitmap());
        return pImage;
    }
};

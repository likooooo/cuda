#pragma once
//== turn debug on in test framework
#ifdef NDEBUG 
#   undef NDEBUG 
#endif
#include <complex>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <memory>
#include <fstream>
#include <type_traist_notebook/type_traist.hpp>
#include <py_helper.hpp>
#include "cuda_vector.hpp"

template<class V1, class V2> inline auto compare_result(const V1& a, const V2& b){
    using real = real_t<typename V1::value_type>;
    const real threshold = std::pow(real(0.1), std::numeric_limits<real>::digits10);
    if(a == b) return std::tuple<real, real, std::vector<real>>();
    std::vector<real> eps(a.size());
    std::transform(a.begin(), a.end(), b.begin(), eps.begin(),[](const auto a, const auto b){return std::abs((a - b) / a);});
    const real max_eps = *std::max_element(eps.begin(), eps.end());
    return std::make_tuple(max_eps, threshold, eps);
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
            (uint8_t)is_complex_v<T> * 4;
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


namespace std
{
    template<> struct numeric_limits<std::complex<float>>{
        constexpr static auto digits10 = 3;//numeric_limits<float>::digits10; 
    };
    template<> struct numeric_limits<std::complex<double>>{
        constexpr static auto digits10 = numeric_limits<double>::digits10; 
    };
}
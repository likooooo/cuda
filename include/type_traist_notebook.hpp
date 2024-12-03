#pragma once 
#include <type_traits>
#include <iostream>
#include <string>
#include <cxxabi.h>
#include <string>
#include <map>
#include <vector>
#include <tuple>
#include <iomanip>
#include <complex>

template<class T> struct unreachable_constexpr_if{unreachable_constexpr_if(){static_assert(std::is_same_v<T, void>, "template unreachable");}}; 

namespace std{
    template<class ...T> struct is_complex: std::false_type{};
    template<class T> struct is_complex <std::complex<T>>: std::true_type{};
    template<class T> constexpr static bool is_complex_v = is_complex<T>::value; 
}

template <class T> struct complex_type { 
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_complex_v<T>);
    using type = std::complex<T>;
};
template <class T> struct real_type {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_complex_v<T>);
    using type = T;
};

template <class T> using complex_t = typename complex_type<T>::type;
template <class T> using real_t = typename real_type<T>::type;
template<class T> constexpr static bool is_complex_v = std::is_same_v<T, complex_t<T>>;
template<class T> constexpr static bool is_real_v = std::is_same_v<T, real_t<T>>;
// template specialization
template <class T> struct real_type<std::complex<T>> { using type = T;};
template <class T> struct complex_type<std::complex<T>> {using type = std::complex<T>;};

//==
template<class T> using remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;
//== is_tuple_v
template <class T>struct is_tuple : std::false_type {};
template <class... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template<class T> constexpr bool is_tuple_v = is_tuple<T>::value;
//== concat tuple
template <class... Args1, class... Args2>
inline auto concat_tuple2(std::tuple<Args1...> arg1, std::tuple<Args2...> arg2) {
    return std::make_tuple(Args1{}..., Args2{}...);
}

template<class TDerived>struct ForkTask;
template<class ...TTask> struct ZipTask;
template<class... Ts> struct check_task_flow_valid;
template<class Task> struct BackwardTask;

//==
template<int N, class ...Task> struct select_task {
    static_assert(N < sizeof...(Task), "out of range");
    using type = std::tuple_element_t<N, std::tuple<Task...>>;
}; 
template<int N, class ...Task> struct select_task<N, std::tuple<Task...>> : public select_task<N, Task...>{};
template<int N, class ...Task> struct select_task<N, ZipTask<Task...>> : public select_task<N, Task...>{};
template<int N, class ...Task> using select_task_t = typename select_task<N, Task...>::type;
template<class ...Task> struct select_last_task {using type = typename select_task<sizeof...(Task) - 1, Task...>::type;}; 
template<class ...Task> using select_last_task_t = typename select_last_task<Task...>::type;
//== struct assign
namespace details{
    template<class TStruct, class TTuple, std::size_t... I>
    TStruct _ToStruct(TTuple&& t,  std::index_sequence<I...>)
    {
        return TStruct{std::forward<decltype(std::get<I>(t))>(std::get<I>(t))...};
    }
}
template<class TStruct, class ...Args>
TStruct ToStruct(const std::tuple<Args...>& t){
    return details::_ToStruct<TStruct>(t, std::make_index_sequence<sizeof...(Args)>{});
}
template<class TStruct, class ...Args>
void ToStruct(TStruct& s, const std::tuple<Args...>& t){
    s = ToStruct<TStruct>(t);
}
//== type to string
template<class T> inline std::string TypeReflection() {
    const char* mangled_name = typeid(T).name();
    int status = -1;
    char* realname = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
    if (status == 0 && realname != nullptr) {
        std::string result(realname);
        std::free(realname);
        return result;
    } else {
        return mangled_name;
    }
}
//==
template<class T, class TAlloc> inline std::ostream& operator<<(std::ostream& out, const std::vector<T, TAlloc>& vec)
{
    std::ostringstream oss;
    oss << "[";
    if(vec.size()){
        for(int i = 0; i < vec.size() - 1; i++) oss << vec[i] << ",";
        oss << vec.back();
    }
    oss <<"]";
    out << oss.str();
    return out;
}
namespace details{
    template<typename TTuple, size_t... Is> inline std::ostream& _to_stream(std::ostream& out, const TTuple& t, std::index_sequence<Is...>, const std::string& separator = ", ")
    {
        constexpr int N = sizeof...(Is);
        if constexpr(N == 0){
            out << std::get<0>(t);
        }
        else if constexpr(N > 0){
            ((out << std::get<Is>(t) << separator), ...);
            out << std::get<N>(t);
        }
        return out;
    }
}
template<class ...Args> inline std::ostream& operator<<(std::ostream& out, const std::tuple<Args...>& t)
{
    using TTuple = std::tuple<Args...>;
    using namespace details;
    out << "(";
    _to_stream(out, t,  std::make_index_sequence<std::tuple_size_v<TTuple> - 1>{});
    out << ")";
    return out;
}
namespace details{
    inline size_t get_invisable_length(const std::string& value){
        int length = 0;
        bool inEscapeSequence = false;
        for (char ch : value) {
            if (inEscapeSequence) {
                if (ch == 'm') {
                    inEscapeSequence = false;
                }
                length++;
            } else {
                if (ch == '\033' || ch == '\x1b') {
                    // 开始一个ANSI转义序列
                    inEscapeSequence = true;
                    length++;
                } 
            }
        }
        return length;
    }
    template<typename T> inline  size_t get_visible_length(const T& value) {
        std::ostringstream oss;
        oss << value;
        return oss.str().length() - get_invisable_length(oss.str());
    }
    template<typename T> inline  void compute_column_width(const T& value, size_t& max_width, size_t truncate_width) {
        max_width = std::min(truncate_width, std::max(max_width, get_visible_length(value)));
    }
    template<typename Tuple, size_t... Is> inline   void compute_column_widths(Tuple&& tuple, std::index_sequence<Is...>, std::vector<size_t>& column_widths, size_t truncate_width) {
        (compute_column_width(std::get<Is>(tuple), column_widths[Is], truncate_width), ...);
    }
    template<class T>inline std::string truncate_with_ellipsis(const T& t, size_t max_length, size_t ellipsis_length = 3) {
        std::ostringstream oss; oss <<t;
        std::string str = oss.str(); 
        if (str.length() <= max_length) {
            return str;
        }
        size_t half_max_length = (max_length - ellipsis_length) / 2;
        if (half_max_length * 2 + ellipsis_length != max_length) {
            half_max_length++; 
        }
        std::string truncated = str.substr(0, half_max_length) + std::string(ellipsis_length, '.')  + str.substr(str.length() - half_max_length);
        return truncated;
    }
    template<typename Tuple, size_t... Is>inline  auto to_str_tuple(const Tuple& t,  const size_t truncate_width,std::index_sequence<Is...>){
        return std::make_tuple(truncate_with_ellipsis(std::get<Is>(t), truncate_width)...);
    }
    template<typename Tuple, size_t... Is> inline  void print_row(Tuple&& tuple, std::index_sequence<Is...>, const std::vector<size_t>& column_widths, size_t truncate_width) {
        auto processed_tuple = std::make_tuple(
            truncate_with_ellipsis(std::get<Is>(tuple), truncate_width)...
        );
        ((std::cout << std::setw(column_widths[Is] + get_invisable_length(std::get<Is>(processed_tuple)))
                    << std::left << std::get<Is>(processed_tuple) << "  "), ...);
        std::cout << std::endl;
    }
    template<typename... Args> inline std::vector<size_t> cal_column_width(const std::vector<std::tuple<Args...>>& lines, const std::array<std::string, sizeof...(Args)>& titles, size_t truncate_width = 50)
    {
        using namespace details;
        //== cal aligin size
        std::vector<size_t> column_widths(sizeof...(Args));
        compute_column_widths(titles, std::make_index_sequence<sizeof...(Args)>(), column_widths, truncate_width);
        for (const auto& line : lines) {
            compute_column_widths(line, std::make_index_sequence<sizeof...(Args)>(), column_widths, truncate_width);
        }
        return column_widths;
    }
    template<class TContainer>inline  void print_column_names(const std::vector<size_t>& column_widths, const TContainer& titles){
        for (size_t i = 0; i < column_widths.size(); ++i) {
            std::cout << std::setw(column_widths[i]) << std::left << titles.at(i) <<"  ";
        }
    }
    inline  void print_split_line (const std::vector<size_t>& column_widths){
        for (size_t i = 0; i < column_widths.size(); ++i) {
            std::cout << std::string(column_widths[i] + 2, '-') << "-";
        }
    };
}
template<typename... Args> inline  std::vector<size_t> print_table(const std::vector<std::tuple<Args...>>& lines, const std::array<std::string, sizeof...(Args)>& titles, size_t truncate_width = 50) {
    using namespace details;
    auto column_widths = cal_column_width(lines, titles, truncate_width);
    print_column_names(column_widths, titles); std::cout << std::endl;
    print_split_line(column_widths); std::cout << std::endl;
    for (const auto& line : lines) {
        print_row(line, std::make_index_sequence<sizeof...(Args)>(), column_widths, truncate_width);
    }
    return column_widths;
}

template <class T> inline std::string AnyTypeToString(T&& input){
    std::stringstream ss;
    ss << input;
    return ss.str();
}
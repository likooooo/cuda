#pragma once
#include <iostream>
#include <string>
#include <type_traits>
#include <cstdlib>
#include <cxxabi.h>
#include <vector>
#include <functional>
#include "util/errorcode.h"
#include "util/Perf.h"
#include "model/Images.h"
#include "model/PyCallback.h"

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

//== debug
template<class T>inline std::string TypeReflection() {
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


template<class ...T> struct is_zip_task : std::false_type{};
template<class ...T> struct is_zip_task<ZipTask<T...>> : std::true_type{};
template<class ...T> constexpr static bool is_zip_task_v = is_zip_task<T...>::value; 
template<class ...T> struct is_fork_task : std::false_type{};
template<class ...T> struct is_fork_task<ForkTask<T...>> : std::true_type{};
template<class ...T> constexpr static bool is_fork_task_v = is_fork_task<T...>::value;

template<class Task>  using backward_return_type = std::invoke_result_t<decltype(&Task::backward), Task*, typename Task::output_type>;
template<class Task> struct is_backward_task : std::false_type{};
template<class Task> struct is_backward_task<BackwardTask<Task>> : std::true_type{};
template<class Task> static constexpr bool is_backward_task_v = is_backward_task<Task>::value; 


namespace details{
    template<class ...T> struct InnerArgs
    {
        using type = std::tuple<remove_cv_ref_t<T>...>;
        static type package_args(T&& ...args){return std::make_tuple(std::forward<T>(args)...);}
    };
    template<class ...T> struct InnerArgs<std::tuple<T...>>
    {
        using type = std::tuple<T...>;
        static type package_args(type&& args){return std::forward<type>(args);}
    };
    template<class ...T> inline auto cal_package_args(T&& ...args){
        return InnerArgs<T...>::package_args(std::forward<T>(args)...);
    } 
}
//== var adpter
using EmptyVar = std::tuple<>;
template<class TVar> static constexpr bool var_valid_v = !std::is_void_v<TVar> && !std::is_same_v<TVar, EmptyVar> && !std::is_same_v<TVar, std::tuple<>>;
template<class TDervied, class = void, class = void> struct TaskVarAdapter{
    using var_type =  EmptyVar;
    void UpdateVar(const var_type& t) const {}
};
//== default  var adapter
template<class TDervied, class TVar> struct TaskVarAdapter<TDervied,  TVar, std::enable_if_t<var_valid_v<TVar>>> : public TVar
{
    using var_type =  TVar;
    void UpdateVar(const TVar& t){static_cast<TVar&>(*this) = t;}
};
//== fork var adapter
template<class TDervied> struct TaskVarAdapter<ForkTask<TDervied>, void, std::enable_if_t<var_valid_v<typename TDervied::var_type>>>
{
    using var_type =  std::vector<typename TDervied::var_type>;
    var_type varvar;
    void UpdateVar(const var_type& var)
    {
        auto& forked_task = static_cast<ForkTask<TDervied>&>(*this).forked_task;
        forked_task.clear(); forked_task.resize(var.size());
        for(int i = 0; i < var.size(); i++) forked_task.at(i).UpdateVar(var.at(i));
    }
};
namespace details
{
    template<class TTuple, class Task, class ...Rest> constexpr static inline auto __concat_all_var_table(TTuple t = TTuple{})
    {
        if constexpr(0 < sizeof...(Rest)){
            if constexpr(var_valid_v<typename Task::var_type>){
                using T = decltype(concat_tuple2(t, std::make_tuple(typename Task::var_type{})));
                return __concat_all_var_table<T, Rest...>();
            }else{
                return __concat_all_var_table<TTuple, Rest...>();
            }
        }else{
            
            if constexpr(var_valid_v<typename Task::var_type>){
                using T = decltype(concat_tuple2(t, std::make_tuple(typename Task::var_type{})));
                if constexpr(1 == std::tuple_size_v<T>)
                    return std::get<0>(T{});
                else
                    return T{};
            }else{
                if constexpr(1 == std::tuple_size_v<TTuple>)
                    return std::get<0>(TTuple{});
                else
                    return TTuple{};
            }
        }
    }
}
//== zip var adapter
template<class ...Task> struct TaskVarAdapter<ZipTask<Task...>, void, std::enable_if_t<(var_valid_v<typename Task::var_type> || ...)>>
{
    using var_type = decltype(details::__concat_all_var_table<std::tuple<>, Task...>());

    void UpdateVar(const var_type& var){
        auto args = details::cal_package_args(var);
        UpdateVarImpl(static_cast<ZipTask<Task...>&>(*this).task, args);
    }
private:
    template<class TTuple,int I = 0, int J = 0> static void UpdateVarImpl(std::tuple<Task...>& tasks, const TTuple& var){
        if constexpr(I < sizeof...(Task) && J < std::tuple_size_v<TTuple>){
            using T1 = typename select_task_t<I, Task...>::var_type;
            using T2 = std::tuple_element_t<J, TTuple>;

            // std::cout <<"var type" << TypeReflection<T1>() << std::endl;
            // std::cout <<"tuple type"<< TypeReflection<T2>() << std::endl;

            if constexpr(std::is_same_v<remove_cv_ref_t<T1>, remove_cv_ref_t<T2>>){
                std::get<I>(tasks).UpdateVar(std::get<J>(var));
                UpdateVarImpl<TTuple, I + 1, J + 1>(tasks, var);
            }else{
                UpdateVarImpl<TTuple, I + 1, J>(tasks, var);
            }
        }else{
            // static_assert(std::tuple_size_v<TTuple> == J, "input-var size  != var-type size"); 
        }
    } 
};

//== meta adapter
template<class TDerived, class TOutput = void, class TVar = void> struct TaskMetaAdapter;
namespace details{
    template<class TOutput, class = void> struct OutputAdapter{ 
        using output_type = void;
        // constexpr static bool enable_result = false;
        void Output()const{}
    };
    template<class TOutput> struct OutputAdapter<TOutput, std::enable_if_t<!is_zip_task_v<TOutput> && !is_fork_task_v<TOutput> && !std::is_void_v<TOutput>>>{ 
        using output_type = TOutput;
        // constexpr static bool enable_result = true;
        output_type output;
        const output_type& Output()const{return output;}
        output_type& Output(){return output;}
    };
}
template<class TDerived, class TOutput, class TVar> struct TaskMetaAdapter : 
    public TaskVarAdapter<TDerived, TVar>, details::OutputAdapter<TOutput>
{
    static std::string& TaskName(){
        static std::string name{TypeReflection<TDerived>()};
        return name;
    }
    void Construct(){} 
    void Destruct(){}
    TaskMetaAdapter()
    {
        static bool print_enable = false;
        if(print_enable){
            std::cout << std::endl;
            std::vector<std::tuple<std::string, std::string>> msg{
                std::make_tuple(std::string("task   name:"), TaskName()), //std::make_tuple(std::string("type"), TypeReflection<TDerived>()),
                std::make_tuple(std::string("varvar type:"), TypeReflection<typename TDerived::var_type>()),
                std::make_tuple(std::string("output type:"), TypeReflection<typename TDerived::output_type>()),
            };
            print_table(msg, {"* Compile information", ""}, 80);
            print_enable = !print_enable;
        }
    }
};
namespace details{
    template<class ...T> struct OutputAdapter<ZipTask<T...>, std::enable_if_t<!std::is_void_v<typename std::tuple_element_t<sizeof...(T) - 1, std::tuple<T...>>::output_type>>>
    { 
        using output_type = typename std::tuple_element_t<sizeof...(T) - 1, std::tuple<T...>>::output_type;
        auto Output()const ->decltype(select_last_task_t<const T...>{}.Output())
        {
            const auto& task = static_cast<const ZipTask<T...>&>(*this).task; 
            return std::get<sizeof...(T) - 1>(task).Output();
        }
        auto Output() ->decltype(select_last_task_t<T...>{}.Output())
        {
            auto& task = static_cast<ZipTask<T...>&>(*this).task; 
            return std::get<sizeof...(T) - 1>(task).Output();
        }
    };
    template<class Task> struct OutputAdapter<ForkTask<Task>, std::enable_if_t<!std::is_void_v<typename Task::output_type>>>
    { 
        using output_type =  std::vector<typename Task::output_type>;
        output_type Output()const
        {
            const auto& forked_task = static_cast<const ForkTask<Task>&>(*this).forked_task;
            output_type out;
            std::transform(forked_task.begin(), forked_task.end(), std::back_inserter(out), [](const auto& t){return t.Output();});
            return out;
        }
    };
}

//== basic task
namespace details
{
    template<class Task> inline void BindForwardContext(Task& t);
    template<class TTuple, size_t I = 0, size_t J = 0> inline void UpdateZipTaskBinding(TTuple& t)
    {
        if constexpr (I < std::tuple_size_v<TTuple>){
            using Task = std::tuple_element_t<I, TTuple>;
            constexpr bool is_bacward = is_backward_task_v<Task>;

            if constexpr(is_zip_task_v<Task> || is_fork_task_v<Task>){
                BindForwardContext(std::get<I>(t));
            } 
            else  if constexpr(is_bacward) {
                std::get<I>(t).forward = &std::get<I - J*2 - 1>(t);
                UpdateZipTaskBinding<TTuple, I + 1, J + 1>(t);
            }else{
                UpdateZipTaskBinding<TTuple, I + 1, J>(t);
            }
        }
    }
    template<class Task> inline  void BindForwardContext(Task& t)
    { 
        if constexpr(is_zip_task_v<Task>) UpdateZipTaskBinding(t.task);
        else if  constexpr(is_fork_task_v<Task>){
            for(auto& sub_task : t.forked_task) BindForwardContext(sub_task);
        }
    }
}
template<class ...T> struct ZipTask :  public TaskMetaAdapter<ZipTask<T...>, ZipTask<T...>>
{
    static_assert(check_task_flow_valid<T...>::value);

    std::tuple<T...> task;
    ZipTask(){
        BindForwardContext(*this);
    }

    template<class ...TArgs> ErrorCode operator()(TArgs&& ...input){
        return operator_impl<0>(details::cal_package_args(std::forward<TArgs>(input)...));
    }
    template<int N, class TTuple> ErrorCode operator_impl(TTuple&& input)
    {
        if constexpr((N + 1) < sizeof...(T)){
            ErrorCode n = std::apply(std::get<N>(task), std::forward<TTuple>(input));
            if(RET_OK != n) return n;
            return operator_impl<N + 1>(details::cal_package_args(std::get<N>(task).Output()));
        }else{
            return std::apply(std::get<N>(task), std::forward<TTuple>(input));
        }
    }
    template<int I> auto& GetFrame(){return std::get<I>(task);}
    template<int I> const auto& GetFrame() const {return std::get<I>(task);}
};
template<class Task>struct ForkTask : public TaskMetaAdapter<ForkTask<Task>, ForkTask<Task>, void>{
    std::vector<Task> forked_task{std::vector<Task>(1)};    
    ForkTask()
    {
        this->TaskName() = "Varvar";
    }
    template<class ...T>ErrorCode operator()(T&& ...args)
    {
        ErrorCode code = RET_ERR;
        int i = 0;  for(auto& t : forked_task)
        {
            code = std::apply(t,  details::cal_package_args(std::forward<T>(args)...));
            if(code != RET_OK) return (ErrorCode)GetForkableErrorCode(code, i);
            i++;
        }
        return code;
    }
    size_t ForkedTaskSize()const{return forked_task.size();}
};

template<class Task> struct BackwardTask  : public TaskMetaAdapter<BackwardTask<Task>, backward_return_type<Task>>
{
    using input_type = typename Task::output_type;
    Task* forward{nullptr};
    ErrorCode operator()(input_type&& in){
        if(nullptr == forward) {
            IMAGE_LOG(WARN, "backward-task need bind forward's (%s) context", Task::TaskName().c_str());
            return RET_INVALID_PARAM;
        }
        this->Output() = forward->backward(std::forward<input_type>(in));
        PyCallback::Call<void>("CheckBackwardImage", __FUNCTION__, *(this->Output()));
        return RET_OK;
    }
};

namespace details{
    template<typename... Ts> struct BackwardTaskHelper 
    {
        using forward_type = std::tuple<Ts...>;

        template<size_t ...I> static auto append_backward(std::index_sequence<I...>)
        {
            using inv_tuple_type = std::tuple<BackwardTask<std::tuple_element_t<sizeof...(I) - 1 - I, forward_type>>...>;
            return std::tuple_cat(std::tuple<Ts...>{}, inv_tuple_type{});
        }
        using tuple_type = decltype(append_backward(std::make_index_sequence<sizeof...(Ts)>{}));

        template<class ...T>static auto task_with_backward(std::tuple<T...>){ return ZipTask<T...>{};}
        using type = decltype(task_with_backward(tuple_type{}));
    };
}
template<class ...Task> using ZipTaskWithBackward = typename details::BackwardTaskHelper<Task...>::type;

/////////////////////////////////////////////////////////////////////////////
template<class T> struct YieldTask : TaskMetaAdapter<YieldTask<T>>{
    ErrorCode operator()(T&& t) {return RET_OK;}
}; 

/////////////////////////////////////////////////////////////////////////////

template<class... Ts>
struct check_task_flow_valid {
    using tuple_type = std::tuple<Ts...>;
    static_assert(sizeof...(Ts) > 0, "At least one task is required.");

    template <class T1, class T2, class = void>
    struct are_compatible : std::false_type {};

    template <class T1, class T2>
    struct are_compatible< T1, T2,
        std::void_t<decltype(std::apply(std::declval<T2>(), std::make_tuple(std::declval<T1>().Output())))>
    > : std::true_type {};

    template <class T, class = void> struct are_compatible_void_check : std::false_type{};
    template <class T> struct are_compatible_void_check<T, std::void_t<decltype(std::declval<T>()())>> : std::true_type {};

    template <std::size_t ...I> static constexpr bool check_helper(std::index_sequence<I...>) {
        constexpr bool value = 
            (
                (
                    // == T2(T1::output())
                    are_compatible<
                        std::tuple_element_t<I, tuple_type>,
                        std::tuple_element_t<I + 1, tuple_type>
                    >::value || 
                    (
                        //== void == T1::output() && T2(void)
                        std::is_void_v<typename std::tuple_element_t<I, tuple_type>::output_type> && 
                        are_compatible_void_check<typename std::tuple_element_t<I + 1, tuple_type>>::value
                    )
                ) && ...
            );
        return value;
    }
    // 检查整个任务流的输入输出要匹配
    static constexpr bool value = check_helper(std::make_index_sequence<sizeof...(Ts) - 1>{});
    static_assert(value, "Task flow is invalid.");
};

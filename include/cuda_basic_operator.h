#pragma once
namespace cuda::vec
{
    template <typename T> void self_add(const int n, const T *x, T *y);
    template <typename T> void self_sub(const int n, const T *x, T *y);
    template <typename T> void self_mul(const int n, const T *x, T *y);
    template <typename T> void self_div(const int n, const T *x, T *y);
}
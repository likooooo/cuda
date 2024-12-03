#pragma once
namespace cuda
{
    template <typename T> void VtAdd(const int n, const T *x, T *y);
    template <typename T> void VtSub(const int n, const T *x, T *y);
    template <typename T> void VtMul(const int n, const T *x, T *y);
    template <typename T> void VtDiv(const int n, const T *x, T *y);
}
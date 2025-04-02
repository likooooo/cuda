#pragma once
namespace cuda
{
    template <typename T> void integral_x(vec2<size_t> shape, T* image);
    template <typename T> void integral_y(vec2<size_t> shape, T* image);
}
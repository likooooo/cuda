#pragma once
#include "cuda_blas.hpp"
#include "cuda_helper.hpp"



inline __both_side__ cuFloatComplex operator +(cuFloatComplex a, cuFloatComplex b)
{
    return cuCaddf(a, b);
}
inline __both_side__ cuDoubleComplex operator +(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCadd(a, b);
}
inline __both_side__ cuFloatComplex operator -(cuFloatComplex a, cuFloatComplex b)
{
    return cuCsubf(a, b);
}
inline __both_side__ cuDoubleComplex operator -(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCsub(a, b);
}

inline __both_side__ cuFloatComplex operator *(cuFloatComplex a, cuFloatComplex b)
{
    return  cuCmulf(a, b);
}
inline __both_side__ cuDoubleComplex operator *(cuDoubleComplex a, cuDoubleComplex b)
{
    return  cuCmul(a, b);
}
inline __both_side__ cuFloatComplex operator /(cuFloatComplex a, cuFloatComplex b)
{
    return cuCdivf(a, b);
}

inline __both_side__ cuDoubleComplex operator /(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCdiv(a, b);
}
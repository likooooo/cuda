#pragma once
#include <iostream>
#include <type_traist_notebook/type_traist.hpp>
// CUDA API error checking
#ifndef CUDA_RT_CALL
#   define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
#ifndef CUDA_RT_LAST_ERROR
#   define CUDA_RT_LAST_ERROR() do { CUDA_RT_CALL(cudaGetLastError()); } while (0)
#endif
// cufft API error chekcing
#ifndef CUFFT_CALL
#   define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

#ifndef CUBLAS_CALL
#   define CUBLAS_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cublasStatus_t>( call );                                                                \
        if ( status != CUBLAS_STATUS_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUBLAS call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUBLAS_CALL

#define __both_side__ __host__ __device__ 

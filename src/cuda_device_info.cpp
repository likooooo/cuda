#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include <type_traits>
#include <cstring>
#include <cuda_helper.hpp>

inline int stringRemoveDelimiter(char delimiter, const char *string) {
  int string_start = 0;

  while (string[string_start] == delimiter) {
    string_start++;
  }

  if (string_start >= static_cast<int>(strlen(string) - 1)) {
    return 0;
  }

  return string_start;
}
inline bool checkCmdLineFlag(const int argc, const char **argv,
                             const char *string_ref) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(
          equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length &&
          !strncasecmp(string_argv, string_ref, length)) {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}
inline int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref) {
  bool bFound = false;
  int value = -1;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        } else {
          value = 0;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound) {
    return value;
  } else {
    return 0;
  }
}
inline const char* ConvertSMVer2ArchName(int major, int minor) {
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
inline int gpuDeviceInit(int devID) {
  int device_count;
  CUDA_RT_CALL(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  CUDA_RT_CALL(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
  CUDA_RT_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  CUDA_RT_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  if (computeMode == cudaComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use cudaSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  CUDA_RT_CALL(cudaSetDevice(devID));
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, ConvertSMVer2ArchName(major, minor));

  return devID;
}
inline int ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    CUDA_RT_CALL(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    CUDA_RT_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    CUDA_RT_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      CUDA_RT_CALL(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
            static_cast<unsigned int>(result), cudaGetErrorName(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}
inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    CUDA_RT_CALL(cudaSetDevice(devID));
    int major = 0, minor = 0;
    CUDA_RT_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    CUDA_RT_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, ConvertSMVer2ArchName(major, minor), major, minor);

  }
  return devID;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int set_cuda_deivce(int argc, char **argv)
{
    int devID;
    cudaDeviceProp props;
    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);
    return devID;
}
inline cudaDeviceProp get_device_properties(int deviceId = 0){
    cudaDeviceProp deviceProp;
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, deviceId));
    return deviceProp;
}
inline const char* get_arch_name(const cudaDeviceProp& deviceProp){
    return ConvertSMVer2ArchName(deviceProp.major, deviceProp.minor);
}
namespace details
{
    template<typename T>
    inline void printDeviceProperty(const std::string& name, T value, const std::string& unit = "") {
        std::cout << std::left << std::setw(30) << name << ": ";
        if constexpr (std::is_same<T, bool>::value) {
            std::cout << (value ? "Yes" : "No");
        }
        else if constexpr (std::is_same<T, std::string>::value) {
            std::cout << value;
        }
        else if constexpr (std::is_arithmetic_v<T>) {
            std::cout << std::fixed << std::setprecision(2) << value;
            if (!unit.empty()) {
                std::cout << " " << unit;
            }
        }
        std::cout << std::endl;
    }

}
inline void print_properties(const cudaDeviceProp& deviceProp)
{
    using namespace details;
    printDeviceProperty<std::string>("Device Name", deviceProp.name);
    printDeviceProperty<std::string>("Arch Name", get_arch_name(deviceProp));
    printDeviceProperty("Major Compute Capability", deviceProp.major);
    printDeviceProperty("Minor Compute Capability", deviceProp.minor);
    printDeviceProperty("Max Threads Per Block", deviceProp.maxThreadsPerBlock);
    printDeviceProperty("Max Threads Dim X/Y/Z", std::to_string(deviceProp.maxThreadsDim[0]) + "x" + std::to_string(deviceProp.maxThreadsDim[1]) + "x" + std::to_string(deviceProp.maxThreadsDim[2]));
    printDeviceProperty("Max Grid Size X/Y/Z", std::to_string(deviceProp.maxGridSize[0]) + "x" + std::to_string(deviceProp.maxGridSize[1]) + "x" + std::to_string(deviceProp.maxGridSize[2]));
    printDeviceProperty("Total Global Memory", static_cast<double>(deviceProp.totalGlobalMem) / (1024 * 1024), "MB");
    printDeviceProperty("Total Constant Memory", static_cast<double>(deviceProp.totalConstMem) / (1024 * 1024), "MB");
    printDeviceProperty("Shared Memory Per Block", static_cast<double>(deviceProp.sharedMemPerBlock) / (1024 * 1024), "MB");
    printDeviceProperty("Reg File Size", deviceProp.regsPerBlock);
    printDeviceProperty("Warp Size", deviceProp.warpSize);
    printDeviceProperty("Memory Pitch", static_cast<double>(deviceProp.memPitch) / (1024 * 1024), "MB");
    printDeviceProperty("Max Registers Per Block", deviceProp.regsPerBlock);
    printDeviceProperty("Clock Rate", static_cast<double>(deviceProp.clockRate) / 1e6, "GHz");
    printDeviceProperty("Total Constant Memory", static_cast<double>(deviceProp.totalConstMem) / (1024 * 1024), "MB");
    printDeviceProperty("Texture Alignment", static_cast<double>(deviceProp.textureAlignment) / (1024 * 1024), "MB");
    printDeviceProperty("Multi-Processor Count", deviceProp.multiProcessorCount);
    printDeviceProperty("Kernel Exec Timeout Enabled", deviceProp.kernelExecTimeoutEnabled);
    printDeviceProperty("Integrated", deviceProp.integrated);
    printDeviceProperty("Can Map Host Memory", deviceProp.canMapHostMemory);
    printDeviceProperty("Compute Mode", deviceProp.computeMode);
    printDeviceProperty("Max Texture 1D Linear", deviceProp.maxTexture1DLinear);
    printDeviceProperty("Async Engine Count", deviceProp.asyncEngineCount);
    printDeviceProperty("Unified Addressing", deviceProp.unifiedAddressing);
    printDeviceProperty("Memory Clock Rate", static_cast<double>(deviceProp.memoryClockRate) / 1e6, "GHz");
    printDeviceProperty("Memory Bus Width", deviceProp.memoryBusWidth, "bits");
    printDeviceProperty("L2 Cache Size", static_cast<double>(deviceProp.l2CacheSize) / (1024 * 1024), "MB");
    printDeviceProperty("Max Threads Per Multi-Processor", deviceProp.maxThreadsPerMultiProcessor);
    printDeviceProperty("Stream Priorities Supported", deviceProp.streamPrioritiesSupported);
    printDeviceProperty("Global L1 Cache Supported", deviceProp.globalL1CacheSupported);
    printDeviceProperty("Local L1 Cache Supported", deviceProp.localL1CacheSupported);
    //printDeviceProperty("Max Shared Memory Per Multi-Processor", static_cast<double>(deviceProp.maxSharedMemoryPerMultiProcessor) / (1024 * 1024), "MB");
    //printDeviceProperty("Managed Memory Supported", deviceProp.managedMemSupported);
    printDeviceProperty("Is Multi-GPU Board", deviceProp.isMultiGpuBoard);
    printDeviceProperty("Multi-GPU Board Group ID", deviceProp.multiGpuBoardGroupID);
    printDeviceProperty("Single to Double Precision Perf Ratio", deviceProp.singleToDoublePrecisionPerfRatio);
    printDeviceProperty("Pageable Memory Access", deviceProp.pageableMemoryAccess);
    printDeviceProperty("Concurrent Kernels", deviceProp.concurrentKernels);
    printDeviceProperty("PCI Domain ID", deviceProp.pciDomainID);
    printDeviceProperty("PCI Bus ID", deviceProp.pciBusID);
    printDeviceProperty("PCI Device ID", deviceProp.pciDeviceID);
    //printDeviceProperty("PCI Device Ordinal", deviceProp.pciDeviceOrdinal);
}
int main(int argc, char** argv) {
    int device_count = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&device_count));
    for (int n = 0; n < device_count; n++) {
        std::cout << "device-" << n <<": \n";
        print_properties(get_device_properties(n));
        std::cout << "\n\n";
    }
}
#ifndef __SW4RAJA_H__
#define __SW4RAJA_H__
#ifdef RAJA03
#include "RAJA/RAJA.hpp"
#else
#include "RAJA/RAJA.hxx"
#endif
using namespace RAJA;
#if defined(CUDA_CODE)
//#include "cuda_runtime.h"
//#include <nvml.h>
//#include <cuda_profiler_api.h>
void CheckError(cudaError_t const err, const char* file, char const* const fun, const int line);
void prefetch_to_device(const float_sw4 *ptr);
#define SW4_CheckDeviceError(err) CheckError(err,__FILE__, __FUNCTION__, __LINE__)
#endif

enum Space { Host, Managed,Device,Pinned,Managed_temps};
void * operator new(std::size_t size,Space loc) throw(std::bad_alloc) ;
void operator delete(void *ptr, Space loc) throw();
void * operator new[](std::size_t size,Space loc) throw(std::bad_alloc) ;
void * operator new[](std::size_t size,Space loc,const char *file,int line);
void operator delete[](void *ptr, Space loc) throw();
void operator delete(void *ptr, Space loc,const char *file, int line) throw();
void operator delete[](void *ptr, Space loc,const char *file, int line) throw();

#endif

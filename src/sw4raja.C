#include "sw4.h"
#include "sw4raja.h"

void * operator new(std::size_t size,Space loc) throw(std::bad_alloc){
#ifdef CUDA_CODE
if (loc==Managed){
  //std::cout<<"Managed allocation \n";
    if (size==0) size=1; // new has to return an valid pointer for 0 size.
    void *ptr;
#ifndef SW4_USE_UMPIRE
    if (cudaMallocManaged(&ptr,size)!=cudaSuccess){
      std::cerr<<"Mananged memory allocation failed "<<size<<"\n";
      throw std::bad_alloc();
    } else {
      //      check_mem();
      // global_variables.curr_mem+=size;
      // global_variables.max_mem=std::max(global_variables.max_mem,global_variables.curr_mem);
      SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,0));
      return ptr;
    }
#else
    umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
    auto allocator = rma.getAllocator("UM_pool");
    ptr = static_cast<void*>(allocator.allocate(size));
    SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,global_variables.device));
    //std::cout<<"PTR 1 "<<ptr<<"\n";
    //SW4_CheckDeviceError(cudaMemset(ptr,0,size));
    return ptr;
#endif
 } else if (loc==Host){
  //std::cout<<"Calling my placement new \n";
  //global_variables.host_curr_mem+=size;
  //global_variables.host_max_mem=std::max(global_variables.host_max_mem,global_variables.host_curr_mem);
  return ::operator new(size);
 } else if (loc==Device){
  //std::cout<<"Managed allocation \n";
    if (size==0) size=1; // new has to return an valid pointer for 0 size.
    void *ptr;
    if (cudaMalloc(&ptr,size)!=cudaSuccess){
      std::cerr<<"Device memory allocation failed "<<size<<"\n";
      throw std::bad_alloc();
    } else return ptr;
 } else if (loc==Pinned){ 
  if (size==0) size=1; // new has to return an valid pointer for 0 size.
  void *ptr;
  SW4_CheckDeviceError(cudaHostAlloc(&ptr,size,cudaHostAllocMapped));
  return ptr;
 } else if (loc==Managed_temps){
#ifdef SW4_USE_UMPIRE
  umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
  auto allocator = rma.getAllocator("UM_pool_temps");
  void *ptr = static_cast<void*>(allocator.allocate(size));
  //SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,0));
  //std::cout<<"PTR 1 "<<ptr<<"\n";
  //SW4_CheckDeviceError(cudaMemset(ptr,0,size));
  return ptr;
#else
  std::cerr<<"Managed_temp location no defined\n";
  return ::operator new(size,Managed); 
#endif
 }
 else {
  std::cerr<<"Unknown memory space for allocation request "<<loc<<"\n";
    throw std::bad_alloc();
  }
// END CUDA CODE
#else
 if (size==0) size=1; // new has to return an valid pointer for 0 size.
 if ((loc==Managed)||(loc==Device)||(loc==Pinned)){
   return ::operator new(size);
  } else if (loc==Host){
    //std::cout<<"Calling my placement new \n";
    return ::operator new(size);
  } else {
    std::cerr<<"Unknown memory space for allocation request\n";
    throw std::bad_alloc();
  }
#endif
}


void * operator new(std::size_t size,Space loc,char *file, int line) throw(std::bad_alloc){
  // std::cout<<"Calling tracking new from "<<line<<" of "<<file<<"\n";
  // pattr_t *ss=new pattr_t;
  // ss->file=file;
  // ss->line=line;
  // ss->type=loc;
  // ss->size=size;
  void *ret= ::operator new(size,loc);
  //patpush(ret,ss);
  return ret;
}

void * operator new[](std::size_t size,Space loc) throw(std::bad_alloc){
#ifdef CUDA_CODE
  if (loc==Managed){
    //std::cout<<"Managed [] allocation \n";
    if (size==0) size=1; // new has to return an valid pointer for 0 size.
    void *ptr;
#ifndef SW4_USE_UMPIRE
    if (cudaMallocManaged(&ptr,size)!=cudaSuccess){
      std::cerr<<"Managed memory allocation failed "<<size<<"\n";
      throw std::bad_alloc();
    } else {
      //check_mem();
      //global_variables.curr_mem+=size;
      //global_variables.max_mem=std::max(global_variables.max_mem,global_variables.curr_mem);
      SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,0));
      return ptr;
    }
#else
    umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
    auto allocator = rma.getAllocator("UM_pool");
    ptr = static_cast<void*>(allocator.allocate(size));
    SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,global_variables.device));
    //std::cout<<"PTR 2 "<<ptr<<"\n";
    return ptr;
#endif
  } else if (loc==Host){
    // std::cout<<"Calling my placement new \n";
    //global_variables.host_curr_mem+=size;
    //global_variables.host_max_mem=std::max(global_variables.host_max_mem,global_variables.host_curr_mem);
    return ::operator new(size);
  } else if (loc==Device){
    //std::cout<<"Managed allocation \n";
    if (size==0) size=1; // new has to return an valid pointer for 0 size.
    void *ptr;
    if (cudaMalloc(&ptr,size)!=cudaSuccess){
      std::cerr<<"Device memory allocation failed "<<size<<"\n";
      throw std::bad_alloc();
    } else return ptr;
  }else if (loc==Pinned){ 
    if (size==0) size=1; // new has to return an valid pointer for 0 size.
    void *ptr;
    SW4_CheckDeviceError(cudaHostAlloc(&ptr,size,cudaHostAllocMapped));
    return ptr;
  }  else if (loc==Managed_temps){
#if defined(SW4_USE_UMPIRE)
    umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
    auto allocator = rma.getAllocator("UM_pool_temps");
    void* ptr = static_cast<void*>(allocator.allocate(size));
    //SW4_CheckDeviceError(cudaMemAdvise(ptr,size,cudaMemAdviseSetPreferredLocation,0));
    //std::cout<<"PTR 1 "<<ptr<<"\n";
    //SW4_CheckDeviceError(cudaMemset(ptr,0,size));
    return ptr;
#else
    std::cerr<<" Memory location Managed_temps is not defined\n";
    return ::operator new(size,Managed); 
#endif
  } else {
    //cudaHostAlloc(&ptr,size+sizeof(size_t)*MEM_PAD_LEN,cudaHostAllocMapped));
    std::cerr<<"Unknown memory space for allocation request "<<loc<<"\n"<<std::flush;
    throw std::bad_alloc();
  }
#else
  return ::operator new(size);
 #endif
}




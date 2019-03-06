#ifdef CUDA_CODE
// typedef NestedPolicy<ExecList<cuda_threadblock_x_exec<16>,cuda_threadblock_y_exec<4>,
//   cuda_threadblock_z_exec<4>>>
//   EXEC_CARTBC;

using EXEC_CARTBC = RAJA::KernelPolicy<
  RAJA::statement::CudaKernelFixed<256,
    RAJA::statement::Tile<0, RAJA::statement::tile_fixed<16>, RAJA::cuda_block_x_loop,
			  RAJA::statement::Tile<1, RAJA::statement::tile_fixed<4>, RAJA::cuda_block_y_loop,
						RAJA::statement::Tile<2, RAJA::statement::tile_fixed<4>, RAJA::cuda_block_z_loop,
								      RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
											   RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
														RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
																     RAJA::statement::Lambda<0> >>>>>>>>;

#define REDUCE_BLOCK_SIZE 256
typedef RAJA::cuda_exec<REDUCE_BLOCK_SIZE> EXEC;
//typedef cuda_reduce<REDUCE_BLOCK_SIZE> REDUCE_POLICY;
using REDUCE_POLICY = RAJA::cuda_reduce;
#define SYNC_DEVICE //cudaDeviceSynchronize();

#else

using EXEC_CARTBC=
  RAJA::KernelPolicy<
  RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
		       RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
					    RAJA::statement::For<2, RAJA::omp_parallel_for_exec,
								 RAJA::statement::Lambda<0>>
		       >
      >
    >;


// using EXEC_CARTBC =
//     RAJA::KernelPolicy<
//       RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
//                                 RAJA::ArgList<0,1,2>,
//         RAJA::statement::Lambda<0>
//       >
//     >;
// typedef NestedPolicy<ExecList<omp_parallel_for_exec,omp_parallel_for_exec,
//   omp_parallel_for_exec>>
//   EXEC_CARTBC;

typedef RAJA::omp_parallel_for_exec EXEC;
typedef omp_reduce REDUCE_POLICY;
#define  SYNC_DEVICE
#endif


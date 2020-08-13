#ifndef _SOLVER_ALGORITHM_H_
#define _SOLVER_ALGORITHM_H_

#include <functional>
#include <future>
#include "MapInstance.h"

#ifdef __CUDACC__
#define __set_block_thread__(...) <<<__VA_ARGS__>>>
#else
#define __set_block_thread__(...)
#define __syncthreads()
#define atomicAdd(ptr, val) 0
#endif

#define __block__ __shared__

namespace c_impl {
  void solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter);
  void solvWayAsync(PixelMap* map, std::function<void(void)> resolve);
}

namespace cuda_impl {
  extern void solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter);
  extern void solvWayAsync(PixelMap* map, std::function<void(void)> resolve);
}

#endif // _SOLVER_ALGORITHM_H_

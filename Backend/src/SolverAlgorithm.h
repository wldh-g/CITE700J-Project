#ifndef _SOLVER_ALGORITHM_H_
#define _SOLVER_ALGORITHM_H_

#include <functional>
#include <future>
#include "MapInstance.h"

namespace c_impl {
  void solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter);
  void solvWayAsync(PixelMap* map, std::function<void(void)> resolve);
}

namespace simd_impl {
  void solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter);
  void solvWayAsync(PixelMap* map, std::function<void(void)> resolve);
}

#endif // _SOLVER_ALGORITHM_H_

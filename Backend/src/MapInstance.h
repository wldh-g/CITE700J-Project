#ifndef _MAP_INSTANCE_H_
#define _MAP_INSTANCE_H_

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <deque>

typedef std::tuple<size_t, size_t> coord_t; // Tuple is generally faster than class structure
typedef uint32_t pixel_t; // Use 32-bit unsigned integer as a pixel value, but this can be changed

class PixelMap {
public:
  PixelMap(const coord_t size, const coord_t start_point, const coord_t end_point,
           const bool* const map, bool* const solution = nullptr,
           pixel_t* const map_proc = nullptr);
  ~PixelMap();

  const coord_t size;
  const size_t pixel_count;

  const coord_t start_point;
  const coord_t end_point;

  const bool* const wall;
  bool* solution;
  pixel_t* proc;

  void reset_solution();
  void reset_proc();

  bool has_solution();
  bool is_finished();
  void mark_as_unsolvable();
  void mark_as_finished();

  pixel_t last_iteration_count;
  double consumed_time;
  double mem_time;
  double slv_time;
  double etc_time;

  bool validate();
  
private:
  bool finished;
  bool solvable;
  bool initialized_in_constructor;
};

#endif // _MAP_INSTANCE_H_

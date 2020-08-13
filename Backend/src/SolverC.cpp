#include <deque>
#include "SolverAlgorithm.cuh"
#include "Timer.h"

void c_impl::solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter) {
  auto [size_x, size_y] = map->size;
  auto [start_x, start_y] = map->start_point;

  CPerfCounter timer;
  timer.Reset();
  timer.Start();

  /*
  * A Poor Iteration Algorithm Explanation
  * 
  * 1. Starting from the start point (add 1 and push to the queue)
  * 2. Propagate the addition to the neighbors in 4 way
  *    a. For each direction, check if next block is a wall
  *    b. If not, add 1 from itself and compare with the next block's value
  *    c. If the next block's value is zero, change next block's value into the added value
  * 3. Do [2] until no more propagation is available
  */

  map->proc[start_x + start_y * size_x] = 1;

  std::deque<coord_t> queue { map->start_point };
  while (queue.size() > 0) {
    auto [target_x, target_y] = queue.front();
    queue.pop_front();
    pixel_t next_value = map->proc[target_x + target_y * size_x] + 1;
    map->last_iteration_count = next_value > map->last_iteration_count ?
      next_value : map->last_iteration_count;

    // Up
    if (size_t next_index = target_x + (target_y - 1) * size_x;
        target_y >= 1 && map->wall[next_index] == false) {
      if (map->proc[next_index] == 0) {
        map->proc[next_index] = next_value;
        queue.push_back(std::tuple { target_x, target_y - 1 });
      }
    }

    // Right
    if (size_t next_index = target_x + 1 + target_y * size_x;
        (target_x + 1) < size_x && map->wall[next_index] == false) {
      if (map->proc[next_index] == 0) {
        map->proc[next_index] = next_value;
        queue.push_back(std::tuple { target_x + 1, target_y });
      }
    }

    // Down
    if (size_t next_index = target_x + (target_y + 1) * size_x;
        (target_y + 1) < size_y && map->wall[next_index] == false) {
      if (map->proc[next_index] == 0) {
        map->proc[next_index] = next_value;
        queue.push_back(std::tuple { target_x, target_y + 1 });
      }
    }

    // Left
    if (size_t next_index = target_x - 1 + target_y * size_x;
        target_x >= 1 && map->wall[next_index] == false) {
      if (map->proc[next_index] == 0) {
        map->proc[next_index] = next_value;
        queue.push_back(std::tuple { target_x - 1, target_y });
      }
    }

    resolveIter();
  }

  timer.Stop();
  map->consumed_time = timer.GetElapsedTime() * 1000;

  map->mark_as_finished();
  resolveIter();
}

void c_impl::solvWayAsync(PixelMap* map, std::function<void(void)> resolve) {
  // TODO
}

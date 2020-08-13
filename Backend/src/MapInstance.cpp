#include <algorithm>
#include "MapInstance.h"

PixelMap::PixelMap(const coord_t size, const coord_t start_point, const coord_t end_point,
                   const bool* const map, bool* const solution, pixel_t* const map_proc)
  : size(size), pixel_count(std::get<0>(size) * std::get<1>(size)), start_point(start_point),
    end_point(end_point), wall(map) {
  if (map_proc == nullptr) {
    this->proc = new pixel_t[this->pixel_count];
    this->solution = new bool[this->pixel_count];
    this->initialized_in_constructor = true;
  } else {
    this->proc = map_proc;
    this->solution = solution;
    this->initialized_in_constructor = false;
  }

  this->last_iteration_count = 0;
  this->consumed_time = 0;
  this->mem_time = 0;

  this->finished = false;
  this->solvable = true;
}

PixelMap::~PixelMap() {
  if (this->initialized_in_constructor) {
    delete this->proc;
    delete this->solution;
  }
}

void PixelMap::reset_solution() {
  // I'm not sure about memset with boolean because it's 1 bit but memset unit is 1 byte
  std::fill(this->solution, this->solution + (this->pixel_count - 1), false);
}

void PixelMap::reset_proc() {
  memset(this->proc, 0, sizeof(pixel_t) * this->pixel_count);
  this->last_iteration_count = 0;
  this->consumed_time = 0;
  this->mem_time = 0;
}

bool PixelMap::is_finished() {
  return this->finished;
}

bool PixelMap::has_solution() {
  return this->solvable;
}

void PixelMap::mark_as_finished() {
  this->finished = true;
}

void PixelMap::mark_as_unsolvable() {
  this->solvable = false;
}

bool PixelMap::validate() {
  if (!this->finished) {
    throw new std::logic_error("Pixel map is not solved yet.");
  } else if (!this->solvable) {
    throw new std::logic_error("Pixel map has no solution.");
  } else {
    auto [size_x, size_y] = this->size;
    auto [start_x, start_y] = this->start_point;
    auto [end_x, end_y] = this->end_point;
    if (this->solution[start_x + start_y * size_x] && this->solution[end_x + end_y * size_x]) {
      return true;
    } else {
      return false;
    }
  }
}

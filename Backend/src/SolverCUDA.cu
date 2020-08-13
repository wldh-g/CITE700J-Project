#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <tuple>
#include "SolverAlgorithm.cuh"
#include "Timer.h"

__global__ void solvIterAsync(bool* wall, pixel_t* proc, size_t* tx, size_t* ty, uint32_t* cnt,
                              size_t pixel_count, size_t size_x, size_t size_y) {
  __block__ size_t blk_x[1000];
  __block__ size_t blk_y[1000];
  __block__ size_t blk_i;

  size_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index == 0) {
    blk_i = 0;
    *cnt = 0;
  }

  auto tx_r = tx[index];
  auto ty_r = ty[index];

  if (tx_r + ty_r != 0) {
    //printf("[%d] %d, %d\n", index, tx_r, ty_r);
    pixel_t next_value = proc[tx_r + ty_r * size_x] + 1;

    size_t next_index = tx_r + (ty_r - 1) * size_x;
    if (ty_r >= 1 && wall[next_index] == false) {
      if (proc[next_index] == 0) { // No overlap because they are checking to all the same direction
        proc[next_index] = next_value;
        size_t blk_o = atomicAdd(&blk_i, 1);
        blk_x[blk_o] = tx_r;
        blk_y[blk_o] = ty_r - 1;
      }
    }
    __syncthreads();

    next_index = tx_r + 1 + ty_r * size_x;
    if ((tx_r + 1) < size_x && wall[next_index] == false) {
      if (proc[next_index] == 0) {
        proc[next_index] = next_value;
        size_t blk_o = atomicAdd(&blk_i, 1);
        blk_x[blk_o] = tx_r + 1;
        blk_y[blk_o] = ty_r;
      }
    }
    __syncthreads();

    next_index = tx_r + (ty_r + 1) * size_x;
    if ((ty_r + 1) < size_y && wall[next_index] == false) {
      if (proc[next_index] == 0) {
        proc[next_index] = next_value;
        size_t blk_o = atomicAdd(&blk_i, 1);
        blk_x[blk_o] = tx_r;
        blk_y[blk_o] = ty_r + 1;
      }
    }
    __syncthreads();

    next_index = tx_r - 1 + ty_r * size_x;
    if (tx_r >= 1 && wall[next_index] == false) {
      if (proc[next_index] == 0) {
        proc[next_index] = next_value;
        size_t blk_o = atomicAdd(&blk_i, 1);
        blk_x[blk_o] = tx_r - 1;
        blk_y[blk_o] = ty_r;
      }
    }
    __syncthreads();

    tx[index] = 0;
    ty[index] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
      size_t blk_o = atomicAdd(cnt, blk_i);
      memcpy(tx + blk_o, blk_x, sizeof(size_t) * blk_i);
      memcpy(ty + blk_o, blk_y, sizeof(size_t) * blk_i);
    }
  }
}

__global__ void initData(size_t x, size_t y, size_t width, pixel_t* proc, size_t* fx, size_t* fy) {
  fx[0] = x;
  fy[0] = y;
  proc[x + y * width] = 1;
}

void cuda_impl::solvIterAsync(PixelMap* map, std::function<void(void)> resolveIter) {
  uint32_t remained_max = 1;
  uint32_t* remained_max_cuda;
  cudaMalloc(&remained_max_cuda, sizeof(uint32_t));

  bool* wall_cuda;
  cudaMalloc(&wall_cuda, sizeof(bool) * map->pixel_count);
  cudaMemcpy(&wall_cuda, map->wall, sizeof(bool) * map->pixel_count, cudaMemcpyHostToDevice);

  pixel_t* proc_cuda;
  cudaMalloc(&proc_cuda, sizeof(pixel_t) * map->pixel_count);
  cudaMemset(&proc_cuda, 0, sizeof(pixel_t) * map->pixel_count);

  size_t* x;
  cudaMalloc(&x, sizeof(size_t) * map->pixel_count);
  cudaMemset(x, 0, sizeof(size_t) * map->pixel_count);
  
  size_t* y;
  cudaMalloc(&y, sizeof(size_t) * map->pixel_count);
  cudaMemset(y, 0, sizeof(size_t) * map->pixel_count);

  size_t size_x = std::get<0>(map->size);
  size_t size_y = std::get<1>(map->size);
  size_t start_x = std::get<0>(map->start_point);
  size_t start_y = std::get<1>(map->start_point);

  ::initData __set_block_thread__(1, 1) (start_x, start_y, size_x, proc_cuda, x, y);
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    printf("CUDA initialization failed (%d).\n", result);
  }

  CPerfCounter timer, timer_mem;
  double mem_time = 0;
  timer.Reset();
  timer.Start();

  dim3 blocks, threads;
  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);
  cudaSetDevice(0);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  do {
    size_t px_mem_expectation = sizeof(pixel_t) + sizeof(bool) + sizeof(size_t) * 2;
    size_t px_max_in_block = prop.sharedMemPerBlock / px_mem_expectation;
    size_t thread_size = std::min(std::min(remained_max, (uint32_t)prop.maxThreadsPerBlock),
                                  (uint32_t)px_max_in_block);
    size_t block_width = (size_t)ceil((double)remained_max / (double)thread_size);
    blocks.x = block_width;
    threads.x = thread_size;

    ::solvIterAsync __set_block_thread__(blocks, threads) (wall_cuda, proc_cuda, x, y,
                                                           remained_max_cuda, map->pixel_count,
                                                           size_x, size_y);
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      printf("CUDA operation stopped (%d).\n", result);
    }

    timer_mem.Reset();
    timer_mem.Start();
    cudaMemcpy(&remained_max, remained_max_cuda, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(map->proc, proc_cuda, sizeof(pixel_t) * map->pixel_count, cudaMemcpyDeviceToHost);
    timer_mem.Stop();
    mem_time += timer_mem.GetElapsedTime();

    map->last_iteration_count += 1;
    resolveIter();
  }
  while (remained_max > 0);

  timer.Stop();
  map->consumed_time = timer.GetElapsedTime() * 1000;
  map->mem_time = mem_time * 1000;

  cudaDeviceReset();

  map->mark_as_finished();
  resolveIter();
}

void cuda_impl::solvWayAsync(PixelMap* map, std::function<void(void)> resolve) {

}

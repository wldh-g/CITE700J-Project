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
  __block__ size_t blk_x[1024];
  __block__ size_t blk_y[1024];
  __block__ size_t blk_i;

  size_t index = threadIdx.x + blockDim.x * blockIdx.x;

  auto tx_r = tx[index];
  auto ty_r = ty[index];

  blk_i = 0;
  *cnt = 0;
  __syncthreads();

  if (tx_r + ty_r != 0) {
    pixel_t next_value = proc[tx_r + ty_r * size_x] + 1;
    size_t flag_a = 0, flag_b = 0, flag_c = 0, flag_d = 0;

    size_t next_index = tx_r + (ty_r - 1) * size_x;
    if (ty_r >= 1 && wall[next_index] == false && proc[next_index] == 0) {
      proc[next_index] = next_value;
      flag_a = 1;
    }
    __syncthreads();

    next_index = tx_r + 1 + ty_r * size_x;
    if ((tx_r + 1) < size_x && wall[next_index] == false && proc[next_index] == 0) {
      proc[next_index] = next_value;
      flag_b = 1;
    }
    __syncthreads();

    next_index = tx_r + (ty_r + 1) * size_x;
    if ((ty_r + 1) < size_y && wall[next_index] == false && proc[next_index] == 0) {
      proc[next_index] = next_value;
      flag_c = 1;
    }
    __syncthreads();

    next_index = tx_r - 1 + ty_r * size_x;
    if (tx_r >= 1 && wall[next_index] == false && proc[next_index] == 0) {
      proc[next_index] = next_value;
      flag_d = 1;
    }
    __syncthreads();

    size_t blk_inc = flag_a + flag_b + flag_c + flag_d;
    if (blk_inc > 0) {
      size_t blk_o = atomicAdd(&blk_i, blk_inc);
      if (flag_a) {
        blk_x[blk_o] = tx_r;
        blk_y[blk_o] = ty_r - 1;
      }
      if (flag_b) {
        blk_x[blk_o + flag_a] = tx_r + 1;
        blk_y[blk_o + flag_a] = ty_r;
      }
      if (flag_c) {
        blk_x[blk_o + flag_a + flag_b] = tx_r;
        blk_y[blk_o + flag_a + flag_b] = ty_r + 1;
      }
      if (flag_d) {
        blk_x[blk_o + flag_a + flag_b + flag_c] = tx_r - 1;
        blk_y[blk_o + flag_a + flag_b + flag_c] = ty_r;
      }
    }
  }

  tx[index] = 0;
  ty[index] = 0;
  __syncthreads();

  if (threadIdx.x == 0) {
    size_t blk_o = atomicAdd(cnt, blk_i);
    memcpy(tx + blk_o, blk_x, sizeof(size_t) * blk_i);
    memcpy(ty + blk_o, blk_y, sizeof(size_t) * blk_i);
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
  cudaMemcpy(wall_cuda, map->wall, sizeof(bool) * map->pixel_count, cudaMemcpyHostToDevice);

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

  dim3 blocks, threads;
  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);
  cudaSetDevice(0);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  CPerfCounter timer_sum, timer_core, timer_mem, timer_etc;
  double time_core = 0;
  double time_mem = 0;
  double time_etc = 0;
  timer_sum.Reset();
  timer_sum.Start();

  do {
    timer_etc.Reset();
    timer_etc.Start();

    size_t thread_size = std::min(remained_max, (uint32_t)prop.maxThreadsPerBlock);
    size_t block_width = (size_t)ceil((double)remained_max / (double)thread_size);
    blocks.x = block_width;
    threads.x = thread_size;

    timer_etc.Stop();
    time_etc += timer_etc.GetElapsedTime();

    timer_core.Reset();
    timer_core.Start();

    ::solvIterAsync __set_block_thread__(blocks, threads) (wall_cuda, proc_cuda, x, y,
                                                           remained_max_cuda, map->pixel_count,
                                                           size_x, size_y);
    cudaError_t result = cudaDeviceSynchronize();

    timer_core.Stop();
    time_core += timer_core.GetElapsedTime();

    if (result != cudaSuccess) {
      printf("CUDA operation stopped (%d).\n", result);
    }

    timer_mem.Reset();
    timer_mem.Start();
    cudaMemcpy(&remained_max, remained_max_cuda, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(map->proc, proc_cuda, sizeof(pixel_t) * map->pixel_count, cudaMemcpyDeviceToHost);
    timer_mem.Stop();
    time_mem += timer_mem.GetElapsedTime();

    map->last_iteration_count += 1;
    resolveIter();
  }
  while (remained_max > 0);

  timer_sum.Stop();
  map->consumed_time = timer_sum.GetElapsedTime() * 1000;
  map->slv_time = time_core * 1000;
  map->mem_time = time_mem * 1000;
  map->etc_time = time_etc * 1000;

  cudaDeviceReset();
  cudaFree(wall_cuda);
  cudaFree(proc_cuda);
  cudaFree(x);
  cudaFree(y);

  map->mark_as_finished();
  resolveIter();
}

void cuda_impl::solvWayAsync(PixelMap* map, std::function<void(void)> resolve) {

}

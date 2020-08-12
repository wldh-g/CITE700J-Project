#include <iostream>

using std::cout;
using std::endl;

int main(int argc, const char** argv) {
  #ifdef _PM_C
  cout << "Generic C Pixel Maze Solver." << endl;
  #endif
  #ifdef _PM_CUDA
  cout << "CUDA Pixel Maze Solver." << endl;
  #endif
  #ifdef _PM_SIMD
  cout << "SIMD Pixel Maze Solver." << endl;
  #endif
  return 0;
}

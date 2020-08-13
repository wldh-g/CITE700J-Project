#include <node.h>
#include "SolverInterface.h"

void InitAll(v8::Local<v8::Object> exports) {
  Solver::Init(exports);
}

NODE_MODULE(PM_C, InitAll)

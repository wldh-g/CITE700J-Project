#ifndef _SOLVER_INTERFACE_H_
#define _SOLVER_INTERFACE_H_

#include <node.h>
#include <node_object_wrap.h>
#include "MapInstance.h"
#include "SolverAlgorithm.cuh"

class Solver : public node::ObjectWrap {
public:
  static void Init(v8::Local<v8::Object> exports);

private:
  explicit Solver(const coord_t size, const coord_t start_point, const coord_t end_point,
                  const bool* const map);
  ~Solver();

  static void New(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void OnIteration(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void OnSolved(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void GetMapProc(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void GetLastPerformanceReport(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void SolveWithC(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void SolveWithCUDA(const v8::FunctionCallbackInfo<v8::Value>& args);
  
  PixelMap* map_;
  const bool* const map_raw_;
  v8::Persistent<v8::Function>* on_iteration_;
  v8::Persistent<v8::Function>* on_solved_;
};

#endif // _SOLVER_INTERFACE_H_

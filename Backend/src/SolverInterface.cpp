#include <functional>
#include <thread>
#include <uv.h>
#include "SolverInterface.h"

using namespace v8;

Solver::Solver(const coord_t size, const coord_t start_point, const coord_t end_point,
               const bool* const map) : map_raw_(map), on_iteration_(nullptr), on_solved_(nullptr)
{
  this->map_ = new PixelMap(size, start_point, end_point, map);
}

Solver::~Solver() {
  delete this->map_;
  delete[] this->map_raw_;
  if (this->on_iteration_ != nullptr) delete this->on_iteration_;
  if (this->on_solved_ != nullptr) delete this->on_solved_;
}

void Solver::Init(Local<Object> exports) {
  Isolate* isolate = exports->GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();

  Local<ObjectTemplate> addon_data_tpl = ObjectTemplate::New(isolate);
  addon_data_tpl->SetInternalFieldCount(1);  // 1 field for the Solver::New()
  Local<Object> addon_data = addon_data_tpl->NewInstance(context).ToLocalChecked();

  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New, addon_data);
  tpl->SetClassName(String::NewFromUtf8(isolate, "Solver").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  // Prototype
  NODE_SET_PROTOTYPE_METHOD(tpl, "onIteration", OnIteration);
  NODE_SET_PROTOTYPE_METHOD(tpl, "onSolved", OnSolved);
  NODE_SET_PROTOTYPE_METHOD(tpl, "getMapProc", GetMapProc);
  NODE_SET_PROTOTYPE_METHOD(tpl, "getPerformanceReport", GetLastPerformanceReport);
  NODE_SET_PROTOTYPE_METHOD(tpl, "solveWithC", SolveWithC);
  NODE_SET_PROTOTYPE_METHOD(tpl, "solveWithSIMD", SolveWithSIMD);

  Local<Function> constructor = tpl->GetFunction(context).ToLocalChecked();
  addon_data->SetInternalField(0, constructor);
  exports->Set(context, String::NewFromUtf8(
    isolate, "Solver").ToLocalChecked(), constructor).FromJust();
}

void Solver::New(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new Solver(...)`
    if (args[0]->IsUndefined()) {
      isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Wrong number of arguments").ToLocalChecked()));
      args.GetReturnValue().SetUndefined();
    } else {
      Local<Object> opt = Local<Object>::Cast(args[0]);

      // Get start point
      auto start_arg = Local<Array>::Cast(opt->Get(context, String::NewFromUtf8(
        isolate, "start").ToLocalChecked()).ToLocalChecked());
      auto start_point = std::tuple {
        (size_t)(start_arg->Get(context, 0).ToLocalChecked())->Uint32Value(context).FromMaybe(0),
        (size_t)(start_arg->Get(context, 1).ToLocalChecked())->Uint32Value(context).FromMaybe(0)
      };
      auto end_arg = Local<Array>::Cast(opt->Get(context, String::NewFromUtf8(
        isolate, "end").ToLocalChecked()).ToLocalChecked());
      auto end_point = std::tuple {
        (size_t)(end_arg->Get(context, 0).ToLocalChecked())->Uint32Value(context).FromMaybe(0),
        (size_t)(end_arg->Get(context, 1).ToLocalChecked())->Uint32Value(context).FromMaybe(0)
      };

      // Get pixel map
      auto map_1 = Local<Array>::Cast(opt->Get(context, String::NewFromUtf8(
        isolate, "map").ToLocalChecked()).ToLocalChecked());
      auto map_2 = Local<Array>::Cast(map_1->Get(context, 0).ToLocalChecked());
      bool* temp = new bool[(size_t)map_1->Length() * (size_t)map_2->Length()];
      for (size_t i = 0; i < map_1->Length(); i += 1) {
        auto map_part = Local<Array>::Cast(map_1->Get(context, i).ToLocalChecked());
        for (size_t j = 0; j < map_2->Length(); j += 1) {
          temp[j + i * (size_t)map_2->Length()] = map_part->Get(
            context, j).ToLocalChecked().As<Boolean>()->BooleanValue(isolate);
        }
      }
      auto size = std::tuple { map_2->Length(), map_1->Length() };

      // Return new instance
      Solver* obj = new Solver(size, start_point, end_point, temp);
      obj->map_->reset_proc();
      obj->map_->reset_solution();
      obj->Wrap(args.This());
      args.GetReturnValue().Set(args.This());
    }
  } else {
    // Invoked as plain function `Solver(...)`, turn into construct call.
    const int argc = 1;
    Local<Value> argv[argc] = { args[0] };
    Local<Function> cons = args.Data().As<Object>()->GetInternalField(0).As<Function>();
    Local<Object> result = cons->NewInstance(context, argc, argv).ToLocalChecked();
    args.GetReturnValue().Set(result);
  }
}

void Solver::OnIteration(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());

  if (solver->on_iteration_ != nullptr) {
    delete solver->on_iteration_;
  }

  auto iter_call = Local<Function>::Cast(args[0]);
  solver->on_iteration_ = new Persistent<Function>(isolate, iter_call);

  args.GetReturnValue().SetUndefined();
}

void Solver::OnSolved(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());

  if (solver->on_solved_ != nullptr) {
    delete solver->on_solved_;
  }

  auto iter_call = Local<Function>::Cast(args[0]);
  solver->on_solved_ = new Persistent<Function>(isolate, iter_call);

  args.GetReturnValue().SetUndefined();
}

void Solver::GetMapProc(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());
  auto [size_x, size_y] = solver->map_->size;

  Local<Array> map_copy = Array::New(isolate, size_y);
  for (size_t y = 0; y < size_y; y += 1) {
    Local<Array> map_part = Array::New(isolate, size_x);
    for (size_t x = 0; x < size_x; x += 1) {
      map_part->Set(context, x, Integer::New(isolate, solver->map_->proc[x + y * size_x]));
    }
    map_copy->Set(context, y, map_part);
  }
  Local<Array> map_with_iter = Array::New(isolate, 2);
  map_with_iter->Set(context, 0, Integer::New(isolate, solver->map_->last_iteration_count));
  map_with_iter->Set(context, 1, map_copy);

  args.GetReturnValue().Set(map_with_iter);
}

void Solver::GetLastPerformanceReport(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());

  Local<Array> perf_report = Array::New(isolate, 1);
  perf_report->Set(context, 0, Number::New(isolate, solver->map_->consumed_time));

  args.GetReturnValue().Set(perf_report);
}

void Solver::SolveWithC(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());

  // CAUTION : V8 is single-threaded, DO NOT USE ANY V8 CODE IN THREADINGS

  // How to give it callback? -> Do it using only C++ native code! (use uv.h because it's in node)
  // Fortunately, It doesn't require any mutex because no writing actions outside the worker thread

  uv_loop_t loop;
  uv_loop_init(&loop);

  uv_async_t async;
  async.data = new std::tuple { isolate, solver->on_iteration_ };
  uv_async_init(&loop, &async, [](uv_async_t* handle) -> void {
    auto tpl = *((std::tuple<Isolate*, Persistent<Function>*>*)(handle->data));
    auto on_iteration = Local<Function>::New(std::get<0>(tpl), *std::get<1>(tpl));
    auto context = std::get<0>(tpl)->GetCurrentContext();
    on_iteration->CallAsFunction(context, context->Global(), 0, nullptr);
  });

  uv_work_t work;
  work.data = new std::tuple { solver->map_, &async };

  uv_queue_work(&loop, &work, [](uv_work_t* work) {
    auto tpl = *((std::tuple<PixelMap*, uv_async_t*>*)(work->data));
    c_impl::solvIterAsync(std::get<0>(tpl), [&]() -> void {
      uv_async_send(std::get<1>(tpl));
    });
  }, [](uv_work_t* work, int status){
    uv_close((uv_handle_t*)std::get<1>(*((std::tuple<PixelMap*, uv_async_t*>*)(work->data))),
             NULL);
  });

  uv_run(&loop, UV_RUN_DEFAULT);

  delete async.data;
  delete work.data;

  auto on_solved = Local<Function>::New(isolate, *solver->on_solved_);
  on_solved->CallAsFunction(context, context->Global(), 0, nullptr);

  args.GetReturnValue().SetUndefined();
}

void Solver::SolveWithSIMD(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext();
  Solver* solver = ObjectWrap::Unwrap<Solver>(args.Holder());

  // CAUTION AGAIN : V8 is single-threaded, DO NOT USE ANY V8 CODE IN THREADINGS

  uv_loop_t loop;
  uv_loop_init(&loop);

  uv_async_t async;
  async.data = new std::tuple { isolate, solver->on_iteration_ };
  uv_async_init(&loop, &async, [](uv_async_t* handle) -> void {
    auto tpl = *((std::tuple<Isolate*, Persistent<Function>*>*)(handle->data));
    auto on_iteration = Local<Function>::New(std::get<0>(tpl), *std::get<1>(tpl));
    auto context = std::get<0>(tpl)->GetCurrentContext();
    on_iteration->CallAsFunction(context, context->Global(), 0, nullptr);
  });

  uv_work_t work;
  work.data = new std::tuple { solver->map_, &async };

  uv_queue_work(&loop, &work, [](uv_work_t* work) {
    auto tpl = *((std::tuple<PixelMap*, uv_async_t*>*)(work->data));
    /*simd_impl::solvIterAsync(std::get<0>(tpl), [&]() -> void {
      uv_async_send(std::get<1>(tpl));
    });*/
  }, [](uv_work_t* work, int status){
    uv_close((uv_handle_t*)std::get<1>(*((std::tuple<PixelMap*, uv_async_t*>*)(work->data))),
             NULL);
  });

  uv_run(&loop, UV_RUN_DEFAULT);

  delete async.data;
  delete work.data;

  auto on_solved = Local<Function>::New(isolate, *solver->on_solved_);
  on_solved->CallAsFunction(context, context->Global(), 0, nullptr);

  args.GetReturnValue().SetUndefined();
}

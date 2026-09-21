#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace go2w_terrain_planner_ros {

class TensorRtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override;
};

template <typename T>
struct TensorRtDestroy {
  void operator()(T* object) const {
    if (object != nullptr) {
      object->destroy();
    }
  }
};

class TensorRtEngine {
 public:
  TensorRtEngine();
  ~TensorRtEngine();
  TensorRtEngine(const TensorRtEngine&) = delete;
  TensorRtEngine& operator=(const TensorRtEngine&) = delete;

  void initialize(
      const std::string& onnx_path,
      const std::string& engine_path,
      bool enable_fp16,
      std::size_t workspace_bytes);
  std::array<float, 2> infer(const std::vector<float>& observation);

 private:
  void loadEngine(const std::string& path);
  void buildEngine(
      const std::string& onnx_path,
      const std::string& engine_path,
      bool enable_fp16,
      std::size_t workspace_bytes);
  void prepareBindings();

  TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TensorRtDestroy<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDestroy<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext,
                  TensorRtDestroy<nvinfer1::IExecutionContext>> context_;
  int input_index_{-1};
  int output_index_{-1};
  void* input_device_{nullptr};
  void* output_device_{nullptr};
  cudaStream_t stream_{nullptr};
};

}  // namespace go2w_terrain_planner_ros

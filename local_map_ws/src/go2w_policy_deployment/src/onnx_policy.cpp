#include "go2w_policy_deployment/onnx_policy.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace go2w_policy_deployment {

class OnnxPolicy::Impl {
 public:
  Impl(
      const std::string& model_path,
      const std::size_t expected_input_dimension,
      const int intra_op_threads)
      : environment(ORT_LOGGING_LEVEL_WARNING, "go2w_policy"),
        expected_dimension(expected_input_dimension),
        memory_info(Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault)) {
    if (model_path.empty() || expected_dimension == 0U || intra_op_threads <= 0) {
      throw std::invalid_argument("invalid ONNX policy configuration");
    }
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(intra_op_threads);
    session_options.SetInterOpNumThreads(1);
    session.reset(new Ort::Session(
        environment, model_path.c_str(), session_options));

    if (session->GetInputCount() != 1U || session->GetOutputCount() != 1U) {
      throw std::runtime_error("ONNX policy must have exactly one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name_value = session->GetInputNameAllocated(0U, allocator);
    const auto output_name_value = session->GetOutputNameAllocated(0U, allocator);
    input_name = input_name_value.get();
    output_name = output_name_value.get();
    if (input_name != "policy_observation" || output_name != "velocity_command") {
      throw std::runtime_error(
          "unexpected ONNX tensor names: input=" + input_name +
          ", output=" + output_name);
    }

    const auto input_info = session->GetInputTypeInfo(0U).GetTensorTypeAndShapeInfo();
    const auto output_info = session->GetOutputTypeInfo(0U).GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("ONNX policy input and output must use float32");
    }
    const std::vector<std::int64_t> input_shape = input_info.GetShape();
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    if (input_shape.size() != 2U ||
        input_shape[1] != static_cast<std::int64_t>(expected_dimension)) {
      throw std::runtime_error("ONNX policy input dimension does not match deployment config");
    }
    if (output_shape.size() != 2U || output_shape[1] != 2) {
      throw std::runtime_error("ONNX policy output must have shape [B,2]");
    }
  }

  std::vector<float> infer(const std::vector<float>& observation) {
    if (observation.size() != expected_dimension) {
      throw std::invalid_argument("policy observation has the wrong dimension");
    }
    for (const float value : observation) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("policy observation contains NaN or Inf");
      }
    }
    const std::array<std::int64_t, 2> input_shape{{
        1, static_cast<std::int64_t>(expected_dimension)}};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(observation.data()),
        observation.size(),
        input_shape.data(),
        input_shape.size());
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    auto outputs = session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input,
        1U,
        output_names,
        1U);
    if (outputs.size() != 1U || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX Runtime returned an invalid policy output");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementCount() != 2U) {
      throw std::runtime_error("ONNX Runtime policy output does not contain two values");
    }
    const float* values = outputs[0].GetTensorData<float>();
    if (!std::isfinite(values[0]) || !std::isfinite(values[1])) {
      throw std::runtime_error("ONNX Runtime policy output contains NaN or Inf");
    }
    return {values[0], values[1]};
  }

  Ort::Env environment;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  std::size_t expected_dimension;
  Ort::MemoryInfo memory_info;
  std::string input_name;
  std::string output_name;
};

OnnxPolicy::OnnxPolicy(
    const std::string& model_path,
    const std::size_t expected_input_dimension,
    const int intra_op_threads)
    : impl_(new Impl(
          model_path, expected_input_dimension, intra_op_threads)) {}

OnnxPolicy::~OnnxPolicy() = default;

std::vector<float> OnnxPolicy::infer(
    const std::vector<float>& observation) {
  return impl_->infer(observation);
}

}  // namespace go2w_policy_deployment

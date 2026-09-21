#include "go2w_terrain_planner_ros/tensorrt_engine.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "go2w_terrain_planner_ros/policy_observation_builder.hpp"

namespace go2w_terrain_planner_ros {
namespace {

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TensorRtDestroy<T>>;

void checkCuda(const cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

bool readableFile(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::ifstream stream(path, std::ios::binary);
  return stream.good() && stream.peek() != std::ifstream::traits_type::eof();
}

std::vector<char> readBinaryFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Cannot open TensorRT engine: " + path);
  }
  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    throw std::runtime_error("TensorRT engine is empty: " + path);
  }
  stream.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!stream.read(data.data(), size)) {
    throw std::runtime_error("Failed to read TensorRT engine: " + path);
  }
  return data;
}

}  // namespace

void TensorRtLogger::log(const Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << message << std::endl;
  }
}

TensorRtEngine::TensorRtEngine() = default;

TensorRtEngine::~TensorRtEngine() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  if (input_device_ != nullptr) {
    cudaFree(input_device_);
  }
  if (output_device_ != nullptr) {
    cudaFree(output_device_);
  }
}

void TensorRtEngine::initialize(
    const std::string& onnx_path,
    const std::string& engine_path,
    const bool enable_fp16,
    const std::size_t workspace_bytes) {
  if (!initLibNvInferPlugins(&logger_, "")) {
    throw std::runtime_error("Failed to register TensorRT plugins");
  }
  if (readableFile(engine_path)) {
    loadEngine(engine_path);
  } else {
    if (!readableFile(onnx_path)) {
      throw std::runtime_error(
          "Neither a cached TensorRT engine nor a readable ONNX model was provided");
    }
    buildEngine(onnx_path, engine_path, enable_fp16, workspace_bytes);
  }
  prepareBindings();
}

void TensorRtEngine::loadEngine(const std::string& path) {
  const std::vector<char> data = readBinaryFile(path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    throw std::runtime_error(
        "Failed to deserialize TensorRT engine (engine/runtime version mismatch?): " +
        path);
  }
}

void TensorRtEngine::buildEngine(
    const std::string& onnx_path,
    const std::string& engine_path,
    const bool enable_fp16,
    const std::size_t workspace_bytes) {
  TrtUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    throw std::runtime_error("Failed to create TensorRT builder");
  }
  const std::uint32_t explicit_batch =
      1U << static_cast<std::uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtUniquePtr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(explicit_batch));
  TrtUniquePtr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger_));
  TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config(
      builder->createBuilderConfig());
  if (!network || !parser || !builder_config) {
    throw std::runtime_error("Failed to create TensorRT ONNX build objects");
  }
  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("TensorRT could not parse ONNX model: " + onnx_path);
  }
  if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
    throw std::runtime_error("Planner ONNX must contain exactly one input and one output");
  }
  nvinfer1::ITensor* input = network->getInput(0);
  if (std::string(input->getName()) != "policy_observation") {
    throw std::runtime_error("Planner ONNX input must be named policy_observation");
  }
  const nvinfer1::Dims input_dimensions = input->getDimensions();
  if (input_dimensions.nbDims != 2 ||
      input_dimensions.d[1] != kPolicyObservationDimension) {
    throw std::runtime_error("Planner ONNX input must have shape [B,280045]");
  }
  if (std::string(network->getOutput(0)->getName()) != "velocity_command") {
    throw std::runtime_error("Planner ONNX output must be named velocity_command");
  }

  builder_config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
  if (enable_fp16 && builder->platformHasFastFp16()) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  // TensorRT 8.5 owns optimization profiles through the builder/config API;
  // the interface has neither destroy() nor a public destructor.
  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();
  if (!profile) {
    throw std::runtime_error("Failed to create TensorRT optimization profile");
  }
  const nvinfer1::Dims2 fixed_shape(1, kPolicyObservationDimension);
  if (!profile->setDimensions(
          input->getName(), nvinfer1::OptProfileSelector::kMIN, fixed_shape) ||
      !profile->setDimensions(
          input->getName(), nvinfer1::OptProfileSelector::kOPT, fixed_shape) ||
      !profile->setDimensions(
          input->getName(), nvinfer1::OptProfileSelector::kMAX, fixed_shape) ||
      builder_config->addOptimizationProfile(profile) < 0) {
    throw std::runtime_error("Failed to configure TensorRT input profile");
  }

  TrtUniquePtr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *builder_config));
  if (!serialized) {
    throw std::runtime_error("TensorRT failed to build the planner engine");
  }
  if (!engine_path.empty()) {
    std::ofstream stream(engine_path, std::ios::binary);
    if (!stream) {
      throw std::runtime_error(
          "Cannot create TensorRT engine cache file: " + engine_path);
    }
    stream.write(
        static_cast<const char*>(serialized->data()),
        static_cast<std::streamsize>(serialized->size()));
    if (!stream) {
      throw std::runtime_error(
          "Failed to write TensorRT engine cache file: " + engine_path);
    }
  }

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(
      serialized->data(), serialized->size()));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize newly built TensorRT engine");
  }
}

void TensorRtEngine::prepareBindings() {
  input_index_ = engine_->getBindingIndex("policy_observation");
  output_index_ = engine_->getBindingIndex("velocity_command");
  if (input_index_ < 0 || output_index_ < 0 ||
      !engine_->bindingIsInput(input_index_) ||
      engine_->bindingIsInput(output_index_)) {
    throw std::runtime_error("TensorRT planner bindings are invalid");
  }
  if (engine_->getBindingDataType(input_index_) != nvinfer1::DataType::kFLOAT ||
      engine_->getBindingDataType(output_index_) != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("TensorRT planner bindings must use float32 I/O");
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }
  if (!context_->setBindingDimensions(
          input_index_, nvinfer1::Dims2(1, kPolicyObservationDimension)) ||
      !context_->allInputDimensionsSpecified()) {
    throw std::runtime_error("Failed to set TensorRT planner input dimensions");
  }
  const nvinfer1::Dims output_dimensions =
      context_->getBindingDimensions(output_index_);
  if (output_dimensions.nbDims != 2 || output_dimensions.d[0] != 1 ||
      output_dimensions.d[1] != 2) {
    throw std::runtime_error("TensorRT planner output must have shape [1,2]");
  }
  checkCuda(cudaMalloc(
      &input_device_,
      static_cast<std::size_t>(kPolicyObservationDimension) * sizeof(float)),
      "cudaMalloc(input)");
  checkCuda(cudaMalloc(&output_device_, 2U * sizeof(float)),
            "cudaMalloc(output)");
  checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

std::array<float, 2> TensorRtEngine::infer(
    const std::vector<float>& observation) {
  if (observation.size() != kPolicyObservationDimension) {
    throw std::invalid_argument("Planner observation must have 280045 elements");
  }
  checkCuda(cudaMemcpyAsync(
      input_device_, observation.data(), observation.size() * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync(input)");
  std::vector<void*> bindings(
      static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
  bindings[static_cast<std::size_t>(input_index_)] = input_device_;
  bindings[static_cast<std::size_t>(output_index_)] = output_device_;
  if (!context_->enqueueV2(bindings.data(), stream_, nullptr)) {
    throw std::runtime_error("TensorRT enqueueV2 failed");
  }
  std::array<float, 2> output{{0.0F, 0.0F}};
  checkCuda(cudaMemcpyAsync(
      output.data(), output_device_, output.size() * sizeof(float),
      cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync(output)");
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return output;
}

}  // namespace go2w_terrain_planner_ros

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace go2w_policy_deployment {

class OnnxPolicy {
 public:
  OnnxPolicy(
      const std::string& model_path,
      std::size_t expected_input_dimension,
      int intra_op_threads = 1);
  ~OnnxPolicy();

  OnnxPolicy(const OnnxPolicy&) = delete;
  OnnxPolicy& operator=(const OnnxPolicy&) = delete;

  std::vector<float> infer(const std::vector<float>& observation);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace go2w_policy_deployment

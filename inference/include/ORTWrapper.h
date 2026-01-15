#pragma once

#include <EncoderInterface.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

namespace nlp::inference {

    class ORTWrapper {
        public:
            ORTWrapper(const std::string& model_path);

            std::vector<float> run(const std::vector<encoder::Token>& tokens);

        private:
            Ort::Env env;
            Ort::Session session;
            Ort::MemoryInfo memory_info;

            std::vector<std::string> input_names;
            std::vector<std::string> output_names;

            static std::vector<float> perform_pooling(const std::vector<float>& raw_logits, size_t num_tokens);
            static std::vector<float> calculate_softmax(const std::vector<float>& logits);
    };

} // namespace nlp::inference

#include <iostream>
#include <cmath>
#include <algorithm>
#include <EncoderInterface.h>
#include "ORTWrapper.h"

namespace nlp::inference {

    ORTWrapper::ORTWrapper(
        const std::string& model_path
    ) :
        env(ORT_LOGGING_LEVEL_WARNING, "BERT_Inference"),
        session(nullptr),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model.
        session = Ort::Session(env, model_path.c_str(), session_options);

        // Discover input and output names.
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session.GetInputCount(); i++) {
            input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++) {
            output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        }
    }


    // PUBLIC METHODS --------------------------------------------------------------------------------------------------

    std::vector<float> ORTWrapper::run(const std::vector<encoder::Token>& tokens) {
        size_t seq_len = tokens.size();
        std::vector<int64_t> input_shape = { 1, (int64_t)seq_len };

        std::vector<int64_t> ids, mask, segments;
        ids.reserve(seq_len);
        mask.reserve(seq_len);
        segments.reserve(seq_len);

        for (const auto& t : tokens) {
            ids.push_back(t.id);
            mask.push_back(t.attention_mask);
            segments.push_back(t.segment_id);
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, ids.data(), ids.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, mask.data(), mask.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, segments.data(), segments.size(), input_shape.data(), input_shape.size()));

        std::vector<const char*> input_names_char;
        for (const auto& name : input_names) input_names_char.push_back(name.c_str());

        std::vector<const char*> output_names_char;
        for (const auto& name : output_names) output_names_char.push_back(name.c_str());

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_char.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_char.data(),
            output_names_char.size()
        );

        // float* float_ptr = output_tensors.front().GetTensorMutableData<float>();
        // auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        //
        // size_t total_elements = 1;
        // for (auto dim : output_shape) total_elements *= dim;

        float* float_ptr = output_tensors.front().GetTensorMutableData<float>();
        size_t total_elements = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> raw_results(float_ptr, float_ptr + total_elements);

        return perform_pooling(raw_results, total_elements);
    }


    // PRIVATE METHODS -------------------------------------------------------------------------------------------------

    std::vector<float> ORTWrapper::perform_pooling(const std::vector<float>& raw_logits, size_t num_tokens) {
        const size_t embedding_dim = 384;  // Fixed for all-MiniLM-L6-v2.
        std::vector<float> sentence_embedding(embedding_dim, 0.0f);

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t d = 0; d < embedding_dim; ++d) {
                sentence_embedding[d] += raw_logits[i * embedding_dim + d];
            }
        }

        // Divide by the number of tokens to get the average.
        for (float& val : sentence_embedding) val /= static_cast<float>(num_tokens);

        return sentence_embedding;
    }

    std::vector<float> ORTWrapper::calculate_softmax(const std::vector<float>& logits) {
        std::vector<float> probabilities(logits.size());

        // Find max element.
        float max_logit = *std::max_element(logits.begin(), logits.end());

        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = std::exp(logits[i] - max_logit);
            sum += probabilities[i];
        }

        for (float& p : probabilities) {
            p /= sum;
        }

        return probabilities;
    }

} // namespace nlp::inference

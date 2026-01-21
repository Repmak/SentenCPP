#include <vector>
#include <cmath>
#include <sentenCPP/embedding_utils/VectorMaths.h>

namespace sentencpp::embedding_utils {

    std::vector<float> VectorMaths::mean_pooling(
        const std::vector<std::vector<float>>& token_embeddings,
        const std::vector<tokenizer::Token>& original_tokens
    ) {
        if (token_embeddings.empty()) return {};

        size_t hidden_size = token_embeddings[0].size();
        std::vector<float> sentence_embedding(hidden_size, 0.0f);
        int valid_token_count = 0;

        for (size_t i = 0; i < token_embeddings.size(); ++i) {
            // Only process tokens where attention_mask is 1 (ignore padding).
            if (original_tokens[i].attention_mask == 1) {
                valid_token_count++;
                for (size_t d = 0; d < hidden_size; ++d) {
                    sentence_embedding[d] += token_embeddings[i][d];
                }
            }
        }

        if (valid_token_count > 0) {
            for (float& val : sentence_embedding) val /= static_cast<float>(valid_token_count);
        }
        return sentence_embedding;
    }

    static std::vector<float> min_pooling(
        const std::vector<std::vector<float>>& token_embeddings,
        const std::vector<tokenizer::Token>& original_tokens
    ) {
        if (token_embeddings.empty()) return {};

        size_t hidden_size = token_embeddings[0].size();
        std::vector<float> sentence_embedding(hidden_size, std::numeric_limits<float>::max());
        bool has_valid_token = false;

        for (size_t i = 0; i < token_embeddings.size(); ++i) {
            if (original_tokens[i].attention_mask == 1) {
                has_valid_token = true;
                for (size_t d = 0; d < hidden_size; ++d) {
                    sentence_embedding[d] = std::min(sentence_embedding[d], token_embeddings[i][d]);
                }
            }
        }
        return has_valid_token ? sentence_embedding : std::vector<float>(hidden_size, 0.0f);
    }

    static std::vector<float> max_pooling(
        const std::vector<std::vector<float>>& token_embeddings,
        const std::vector<tokenizer::Token>& original_tokens
    ) {
        if (token_embeddings.empty()) return {};

        size_t hidden_size = token_embeddings[0].size();
        std::vector<float> sentence_embedding(hidden_size, -std::numeric_limits<float>::max());
        bool has_valid_token = false;

        for (size_t i = 0; i < token_embeddings.size(); ++i) {
            if (original_tokens[i].attention_mask == 1) {
                has_valid_token = true;
                for (size_t d = 0; d < hidden_size; ++d) {
                    sentence_embedding[d] = std::max(sentence_embedding[d], token_embeddings[i][d]);
                }
            }
        }
        return has_valid_token ? sentence_embedding : std::vector<float>(hidden_size, 0.0f);
    }

    static float euclidean_distance(
        const std::vector<float>& vec_a,
        const std::vector<float>& vec_b
    ) {
        if (vec_a.size() != vec_b.size()) return -1.0f;

        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < vec_a.size(); ++i) {
            float diff = vec_a[i] - vec_b[i];
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff);
    }

    float VectorMaths::cosine_similarity(
        const std::vector<float>& vec_a,
        const std::vector<float>& vec_b
    ) {
        if (vec_a.size() != vec_b.size()) return 0.0f;

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < vec_a.size(); ++i) {
            dot += vec_a[i] * vec_b[i];
            norm_a += vec_a[i] * vec_a[i];
            norm_b += vec_b[i] * vec_b[i];
        }
        return (norm_a == 0 || norm_b == 0) ? 0.0f : dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    std::vector<float> VectorMaths::calculate_softmax(
        const std::vector<float>& logits
    ) {
        std::vector<float> probabilities(logits.size());

        // Find max element.
        float max_logit = *std::max_element(logits.begin(), logits.end());

        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = std::exp(logits[i] - max_logit);
            sum += probabilities[i];
        }

        for (float& p : probabilities) p /= sum;
        return probabilities;
    }

} // namespace sentencpp::embedding_utils

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <string_view>

namespace sentencpp::tokenizer {

    struct Token {
        int64_t id;              // The numerical ID according to the model's vocabulary. Must be int64_t to be compatible with ORT.
        std::string text;        // The original string representation (not strictly needed by the Onnx model).
        int64_t attention_mask;  // 1 for real tokens, 0 for padding. Must be int64_t to be compatible with ORT.
        int64_t segment_id;      // Defines what sentence the token belongs to. Must be int64_t to be compatible with ORT.

        friend std::ostream& operator<<(std::ostream& os, const Token& instance) {
            os << "id: " << instance.id
            << ", text: \"" << instance.text
            << "\", attention_mask:" << instance.attention_mask
            << ", segment: " << instance.segment_id;
            return os;
        }
    };

    struct TokenizerBaseConfig {
        std::size_t max_input_chars_per_word = 100;
        std::size_t max_length = 128;
        bool to_lowercase = true;
        bool strip_accents = true;
        bool clean_text = true;
        bool handle_chinese_chars = true;
        std::string padding_token = "[PAD]";
        std::string unknown_token = "[UNK]";
        std::string classification_token = "[CLS]";
        std::string separator_token = "[SEP]";
        std::string mask_token = "[MASK]";
    };

    struct WordPieceConfig : public TokenizerBaseConfig {
        std::string config_path;  // Path to the config file. Eg: "tokenizer.json".
        std::string vocab_key = "/model/vocab";  // Path to the vocabulary object within the config file. Eg: "/model/vocab".
    };

    struct BPEConfig : public TokenizerBaseConfig {
        // todo
    };

    struct UnigramConfig : public TokenizerBaseConfig {
        // todo
    };

    class TokenizerInterface {
        public:
            virtual ~TokenizerInterface() = default;

            // Returns the total vocabulary size.
            [[nodiscard]] virtual size_t get_vocab_size() const = 0;

            // Tokenize raw text into Token objects.
            [[nodiscard]] virtual std::vector<Token> tokenize(std::string_view text) const = 0;
    };

} // namespace sentencpp::tokenizer

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>

#include "EncoderInterface.h"

namespace nlp::encoder {

    class WordPiece : public EncoderInterface {
        public:
            WordPiece(
                const std::string& config_path,
                const std::string& vocab_key,
                bool clean_text,
                bool to_lowercase,
                bool strip_accents,
                bool handle_chinese_chars,
                std::size_t max_input_chars_per_word,
                std::size_t max_length,
                std::string padding_token="[PAD]",
                std::string unknown_token="[UNK]",
                std::string classification_token="[CLS]",
                std::string separator_token="[SEP]",
                std::string mask_token="[MASK]"
            );

            [[nodiscard]] std::vector<Token> encode(std::string_view text) const override;

            [[nodiscard]] size_t get_vocab_size() const override { return vocab_list_->size(); }
            [[nodiscard]] const VocabList& get_vocab_list() const { return *vocab_list_; }

        private:
            std::unique_ptr<VocabList> vocab_list_;
            bool clean_text;
            bool to_lowercase;
            bool strip_accents;
            bool handle_chinese_chars;
            std::size_t max_input_chars_per_word;
            std::size_t max_length;

            // Splits text by whitespace and punctuation.
            [[nodiscard]] std::vector<std::string_view> split_text(std::string_view text) const;

            // Encode each word into one or more tokens (using MaxMatch algorithm).
            [[nodiscard]] std::vector<Token> encode_word(std::string_view word) const;

            // Truncation and adding special tokens.
            void post_processing(std::vector<Token>& tokens) const;

            // Normalising user input.
            void clean_text_inplace(std::string& text) const;
            void to_lowercase_inplace(std::string& text) const;
            void strip_accents_inplace(std::string& text) const;
            void handle_chinese_chars_inplace(std::string& text) const;
    };

} // namespace nlp::encoder

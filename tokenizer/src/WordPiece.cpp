#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "WordPiece.h"

using json = nlohmann::json;

namespace nlp::tokenizer {

    WordPiece::WordPiece(
        const std::string& config_path,  // Eg: "tokenizer.json".
        const std::string& vocab_key,  // Eg: "/model/vocab".
        bool lowercase,
        bool clean_text,
        bool handle_chinese_chars,
        std::size_t max_length
    ) : vocab_list_(std::make_unique<VocabList>()) {

        this->lowercase = lowercase;
        this->clean_text = clean_text;
        this->handle_chinese_chars = handle_chinese_chars;
        this->max_length = max_length;

        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open config file: " << config_path << std::endl;
            exit(-1);
        }

        json config;
        try {
            file >> config;
            auto vocab = config.at(json::json_pointer(vocab_key));

            for (auto& [token, id] : vocab.items()) {
                uint32_t token_id = id.get<uint32_t>();
                if (!vocab_list_->set_token(token, token_id)) {
                    std::cerr << "Warning: Could not set token '" << token << "' with ID " << token_id << std::endl;
                }
            }

        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            exit(-1);
        }
    }

    std::vector<Token> WordPiece::encode(std::string_view text) const {


        return {};
    }

    TokenRole WordPiece::identify_special_token(uint32_t id) const {
        return TokenRole::None;
    }

    void WordPiece::build_byte_encoder() {

    }

    std::vector<std::string> WordPiece::bpe_merge(std::string_view word) const {

        return {};
    }

    std::vector<std::string_view> WordPiece::pre_tokenize(std::string_view text) const {

        return {};
    }

} // namespace nlp::tokenizer

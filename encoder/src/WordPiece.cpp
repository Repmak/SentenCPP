#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unicode/utypes.h>
#include <unicode/unistr.h>
#include <unicode/translit.h>
#include <nlohmann/json.hpp>
#include "WordPiece.h"


using json = nlohmann::json;

namespace nlp::encoder {

    WordPiece::WordPiece(
        const std::string& config_path,  // Eg: "tokenizer.json".
        const std::string& vocab_key,  // Eg: "/model/vocab".
        bool clean_text,
        bool to_lowercase,
        bool strip_accents,
        bool handle_chinese_chars,
        std::size_t max_input_chars_per_word,
        std::size_t max_length,
        std::string padding_token,
        std::string unknown_token,
        std::string classification_token,
        std::string separator_token,
        std::string mask_token
    ) : vocab_list_(std::make_unique<VocabList>()) {

        this->clean_text = clean_text;
        this->to_lowercase = to_lowercase;
        this->strip_accents = strip_accents;
        this->handle_chinese_chars = handle_chinese_chars;
        this->max_input_chars_per_word = max_input_chars_per_word;
        this->max_length = max_length;

        vocab_list_->set_special_token(padding_token, TokenRole::Padding);
        vocab_list_->set_special_token(unknown_token, TokenRole::Unknown);
        vocab_list_->set_special_token(classification_token, TokenRole::Classification);
        vocab_list_->set_special_token(separator_token, TokenRole::Separator);
        vocab_list_->set_special_token(mask_token, TokenRole::Mask);

        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open config file: " << config_path << std::endl;
            exit(-1);
        }

        json config;
        try {
            file >> config;
            auto vocab = config.at(json::json_pointer(vocab_key));

            for (auto& [token_str, id] : vocab.items()) {
                int64_t token_id = id.get<int64_t>();
                if (!vocab_list_->set_token(token_str, token_id)) {
                    std::cerr << "Warning: Could not set token '" << token_str << "' with ID " << token_id << std::endl;
                }
            }

            std::vector<std::pair<std::string, std::string>> required_tokens = {
                {padding_token, "Padding"},
                {unknown_token, "Unknown"},
                {classification_token, "Classification"},
                {separator_token, "Separator"},
                {mask_token, "Mask"}
            };

            for (const auto& [token_str, role_name] : required_tokens) {
                if (vocab_list_->token_to_id(token_str) == std::nullopt) {
                    throw std::runtime_error(
                        "Special token '" + token_str + "' (" + role_name + ") not found in vocabulary file."
                    );
                }
            }
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            exit(-1);
        }
    }


    // PUBLIC METHODS --------------------------------------------------------------------------------------------------

    std::vector<Token> WordPiece::encode(std::string_view text) const {
        std::string normalised_text(text);  // Local copy to work with.
        if (clean_text) clean_text_inplace(normalised_text);
        if (to_lowercase) to_lowercase_inplace(normalised_text);
        if (strip_accents) strip_accents_inplace(normalised_text);
        if (handle_chinese_chars) handle_chinese_chars_inplace(normalised_text);

        std::cout << "Normalised text: " << normalised_text << std::endl;

        std::vector<std::string_view> words = split_text(normalised_text);
        std::vector<Token> all_tokens;

        for (const auto& word : words) {
            std::vector<Token> word_tokens = encode_word(word);
            all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
        }

        post_processing(all_tokens);
        return all_tokens;
    }


    // PRIVATE METHODS -------------------------------------------------------------------------------------------------

    std::vector<std::string_view> WordPiece::split_text(std::string_view text) const {
        std::vector<std::string_view> words;
        size_t i = 0;
        const size_t n = text.length();

        while (i < n) {
            // Skip Whitespace.
            if (std::isspace(static_cast<unsigned char>(text[i]))) {
                i++;
                continue;
            }

            // Separate punctuation.
            if (std::ispunct(static_cast<unsigned char>(text[i]))) {
                size_t start = i;
                char punct_char = text[i];

                // Peek ahead to group identical punctuation.
                while (i < n && text[i] == punct_char && std::ispunct(static_cast<unsigned char>(text[i]))) i++;

                words.push_back(text.substr(start, i - start));
                continue;
            }

            if (std::ispunct(static_cast<unsigned char>(text[i]))) {
                words.push_back(text.substr(i, 1));
                i++;
                continue;
            }

            const size_t start = i;
            while (
                i < n &&
                !std::isspace(static_cast<unsigned char>(text[i])) &&
                !std::ispunct(static_cast<unsigned char>(text[i]))
            ) i++;

            words.push_back(text.substr(start, i - start));
        }
        return words;
    }

    std::vector<Token> WordPiece::encode_word(std::string_view word) const {
        std::vector<Token> tokens;
        size_t start = 0;
        const size_t n = word.length();

        std::string unknown_token_str = vocab_list_->get_special_token_val(TokenRole::Unknown);
        int64_t unknown_token_id = vocab_list_->token_to_id(unknown_token_str).value();

        if (n >= max_input_chars_per_word) return {Token{unknown_token_id, std::string(word), 1, 0}};

        while (start < n) {
            size_t end = n;
            std::optional<uint32_t> best_id = std::nullopt;
            std::string best_substr;

            while (start < end) {
                std::string substr(word.substr(start, end - start));

                if (start > 0) substr.insert(0, "##");

                if (auto id = vocab_list_->token_to_id(substr)) {
                    best_id = id;
                    best_substr = std::move(substr);
                    break;
                }
                end--;
            }

            // Entire word is unknown if a match cannot be found.
            if (!best_id.has_value()) return {Token{unknown_token_id, std::string(word), 1, 0}};

            tokens.push_back(Token{best_id.value(), best_substr, 1, 0});
            start = end;
        }
        return tokens;
    }

    void WordPiece::post_processing(std::vector<Token>& tokens) const {
        std::string classification_token_str = vocab_list_->get_special_token_val(TokenRole::Classification);
        int64_t classification_token_id = vocab_list_->token_to_id(classification_token_str).value();
        std::string separator_token_str = vocab_list_->get_special_token_val(TokenRole::Separator);
        int64_t separator_token_id = vocab_list_->token_to_id(separator_token_str).value();
        std::string padding_token_str = vocab_list_->get_special_token_val(TokenRole::Padding);
        int64_t padding_token_id = vocab_list_->token_to_id(padding_token_str).value();

        // Reserve index 0 for [CLS] and index 127 for [SEP].
        if (tokens.size() > (max_length - 2)) {
            tokens.resize(max_length - 2);
            std::cerr << "Warning: Tokens truncated. max_length = " << max_length << std::endl;
        }

        tokens.insert(tokens.begin(), Token{classification_token_id, "", 1, 0});
        tokens.push_back(Token{separator_token_id, "", 1, 0});

        // Add padding if necessary.
        if (tokens.size() < max_length) {
            tokens.reserve(max_length);
            while (tokens.size() < max_length) {
                tokens.push_back(Token{padding_token_id, "", 0, 0});
            }
        }
    }

    void WordPiece::clean_text_inplace(std::string& text) const {
        icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(text);
        icu::UnicodeString cleaned;
        bool last_was_space = false;

        for (int32_t i = 0; i < ustr.length(); ) {
            UChar32 c = ustr.char32At(i);
            int32_t next_i = ustr.moveIndex32(i, 1);
            int8_t category = u_charType(c);

            if (c == 0 || c == 0xfffd || category == U_CONTROL_CHAR || category == U_FORMAT_CHAR) {
                // Skip.
            } else if (u_isUWhiteSpace(c)) {
                if (!last_was_space) {
                    cleaned.append((UChar32)' ');
                    last_was_space = true;
                }
            } else {
                cleaned.append(c);
                last_was_space = false;
            }
            i = next_i;
        }

        text.clear();
        cleaned.toUTF8String(text);
    }

    void WordPiece::to_lowercase_inplace(std::string& text) const {
        std::ranges::transform(text, text.begin(),
            [](unsigned char c) { return std::tolower(c); });
    }

    void WordPiece::strip_accents_inplace(std::string& text) const {
        UErrorCode status = U_ZERO_ERROR;
        icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(text);

        std::unique_ptr<icu::Transliterator> remover(
            icu::Transliterator::createInstance("NFD; [:M:] Remove; NFC", UTRANS_FORWARD, status)
        );

        if (U_SUCCESS(status)) remover->transliterate(ustr);

        text.clear();
        ustr.toUTF8String(text);
    }

    void WordPiece::handle_chinese_chars_inplace(std::string& text) const {
            std::cerr << "Warning: Method handle_chinese_chars_inplace is not implemented" << std::endl;
    }

} // namespace nlp::encoder

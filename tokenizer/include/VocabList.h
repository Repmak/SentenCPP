#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace sentencpp::tokenizer {

    enum class TokenRole { Padding, Unknown, Classification, Separator, Mask };

    class VocabList {
        public:
            VocabList() = default;

            // Add a key value pair to the mappings.
            bool set_token(const std::string& token_str, int64_t token_id);
            bool set_special_token(const std::string& token_str, TokenRole token_role);

            [[nodiscard]] const std::unordered_map<std::string, int64_t>& get_string_to_id_map() const { return string_to_id_map_; }
            [[nodiscard]] const std::vector<std::string>& get_id_to_string_map() const { return id_to_string_map_; }

            [[nodiscard]] const std::unordered_map<TokenRole, std::string>& get_special_tokens_map_() const { return special_tokens_map_; }
            [[nodiscard]] std::string get_special_token_val(const TokenRole token_role) const { return special_tokens_map_.at(token_role); }

            [[nodiscard]] std::optional<int64_t> token_to_id(const std::string& token_str) const;
            [[nodiscard]] std::optional<std::string> id_to_token(int64_t token_id) const;

            [[nodiscard]] size_t size() const { return id_to_string_map_.size(); }
            friend std::ostream& operator<<(std::ostream& os, const VocabList& instance);

        private:
            std::unordered_map<std::string, int64_t> string_to_id_map_;
            std::vector<std::string> id_to_string_map_;
            std::unordered_map<TokenRole, std::string> special_tokens_map_;

            // std::unordered_map<std::string, std::string> special_tokens_map_ = {
            //     {"padding", "[PAD]"},
            //     {"unknown", "[UNK]"},
            //     {"classification", "[CLS]"},
            //     {"separator", "[SEP]"},
            //     {"mask", "[MASK]"}
            // };
    };

} // namespace sentencpp::tokenizer

#include <iostream>
#include <iomanip>
#include "VocabList.h"

namespace nlp::tokenizer {

    bool VocabList::set_token(const std::string& token_str, const int64_t token_id) {
        // Check the token and id.
        if (token_str.empty() || string_to_id_map_.contains(token_str)) return false;

        // Check that we don't overwrite data.
        if (token_id < id_to_string_map_.size() && !id_to_string_map_[token_id].empty()) return false;

        // Ensure vector is large enough.
        if (token_id >= id_to_string_map_.size()) id_to_string_map_.resize(token_id + 1);

        // Set the mappings.
        string_to_id_map_[token_str] = token_id;
        id_to_string_map_[token_id] = token_str;
        return true;
    }

    bool VocabList::set_special_token(const std::string& token_str, TokenRole token_role) {
        if (token_str.empty() || special_tokens_map_.contains(token_role)) return false;
        special_tokens_map_[token_role] = token_str;
        return true;
    }

    std::optional<int64_t> VocabList::token_to_id(const std::string& token_str) const {
        const auto got = string_to_id_map_.find(token_str);
        if (got == string_to_id_map_.end()) return std::nullopt;
        return got->second;
    }

    std::optional<std::string> VocabList::id_to_token(const int64_t token_id) const {
        if (token_id >= id_to_string_map_.size()) return std::nullopt;
        const std::string& token_str = id_to_string_map_[token_id];
        if (token_str.empty()) return std::nullopt;
        return token_str;
    }

    std::ostream& operator<<(std::ostream& os, const VocabList& instance) {
        os << std::left << std::setw(20) << "Token" << " | " << "ID" << "\n";
        os << std::string(30, '-') << "\n";
        for (const auto& [token, id] : instance.string_to_id_map_) {
            os << std::left << std::setw(20) << token << " | " << id << "\n";
        }
        return os;
    }

} // namespace nlp::tokenizer

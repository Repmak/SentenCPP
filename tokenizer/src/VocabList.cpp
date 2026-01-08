#include <iostream>
#include "VocabList.h"

namespace nlp::tokenizer {

    bool VocabList::set_token(const std::string& token, uint32_t id) {
        // Check the token and id.
        if (token.empty() || string_to_id_map_.contains(token)) return false;

        // Check that we don't overwrite data.
        if (id < id_to_string_map_.size() && !id_to_string_map_[id].empty()) return false;

        // Ensure vector is large enough.
        if (id >= id_to_string_map_.size()) id_to_string_map_.resize(id + 1);

        // Set the mappings.
        string_to_id_map_[token] = id;
        id_to_string_map_[id] = token;

        switch (config.get_special_role(token)) {
            case TokenRole::Padding:   special_ids_.padding = id; break;
            case TokenRole::Unknown:   special_ids_.unknown = id; break;
            case TokenRole::Classification:   special_ids_.classification = id; break;
            case TokenRole::Separator:   special_ids_.separator = id; break;
            case TokenRole::Mask:   special_ids_.mask = id; break;
            default: break;
        }
        return true;
    }

    std::optional<uint32_t> VocabList::token_to_id(const std::string& token) const {
        auto got = string_to_id_map_.find(token);
        if (got == string_to_id_map_.end()) return std::nullopt;
        return got->second;
    }

    std::optional<std::string> VocabList::id_to_token(uint32_t id) const {
        if (id >= id_to_string_map_.size()) return std::nullopt;
        const std::string& token = id_to_string_map_[id];
        if (token.empty()) return std::nullopt;
        return token;
    }

} // namespace nlp::tokenizer

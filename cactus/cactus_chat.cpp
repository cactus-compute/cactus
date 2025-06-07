#include "cactus.h"
#include "common.h"
#include "json.hpp"

using json = nlohmann::ordered_json;

namespace cactus {

common_chat_params cactus_context::getFormattedChatWithJinja(
  const std::string &messages,
  const std::string &chat_template,
  const std::string &json_schema,
  const std::string &tools,
  const bool &parallel_tool_calls,
  const std::string &tool_choice
) const {
    if (!model || !templates) {
         LOG_ERROR("Model or templates not loaded, cannot format chat.");
         return {}; 
    }
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
        auto useTools = !tools.empty();
        if (useTools) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools));
        }
        if (!tool_choice.empty()) {
             inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
        }
        if (!json_schema.empty()) {
            inputs.json_schema = json::parse(json_schema);
        }
    } catch (const json::exception& e) {
        LOG_ERROR("JSON parsing error during chat formatting: %s", e.what());
        throw std::runtime_error("Invalid JSON input for chat formatting.");
    }
    inputs.parallel_tool_calls = parallel_tool_calls;
    inputs.extract_reasoning = params.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    if (!chat_template.empty()) {
        try {
            if (!common_chat_verify_template(chat_template.c_str(), true)) {
                 LOG_WARNING("Provided custom Jinja template is invalid.");
            }
            auto tmps = common_chat_templates_init(model, chat_template);
            return common_chat_templates_apply(tmps.get(), inputs);
        } catch (const std::exception& e) {
             LOG_ERROR("Error applying custom chat template: %s", e.what());
              return common_chat_templates_apply(templates.get(), inputs);
        }
    } else {
        return common_chat_templates_apply(templates.get(), inputs);
    }
}

std::string cactus_context::getFormattedChat(
  const std::string &messages,
  const std::string &chat_template
) const {
    if (!model || !templates) {
         LOG_ERROR("Model or templates not loaded, cannot format chat.");
         return ""; 
    }
    common_chat_templates_inputs inputs;
    inputs.use_jinja = false;
     try {
         inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
     } catch (const json::exception& e) {
         LOG_ERROR("JSON parsing error during chat formatting: %s", e.what());
         throw std::runtime_error("Invalid JSON input for chat formatting.");
     }

    if (!chat_template.empty()) {
         try {
             if (!common_chat_verify_template(chat_template.c_str(), false)) {
                 LOG_WARNING("Provided custom standard template is invalid.");
             }
             auto tmps = common_chat_templates_init(model, chat_template);
             return common_chat_templates_apply(tmps.get(), inputs).prompt;
         } catch (const std::exception& e) {
             LOG_ERROR("Error applying custom chat template: %s", e.what());
             return common_chat_templates_apply(templates.get(), inputs).prompt;
         }
    } else {
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }
}

} // namespace cactus 
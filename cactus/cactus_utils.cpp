#include "cactus.h"
#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sstream>

namespace cactus {

bool cactus_verbose = true; 

void log(const char *level, const char *function, int line,
                const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);
        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
             if (!cactus_verbose && strcmp(level, "VERBOSE") == 0) {
                 va_end(args);
                 return;
             }
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "Cactus", prefix, args);
        va_end(args);
    #else
        if (!cactus_verbose && strcmp(level, "VERBOSE") == 0) {
            return;
        }
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}


void llama_batch_clear(llama_batch *batch) {
    if (batch) {
         batch->n_tokens = 0;
    }
}

void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_ids, bool logits) {
     if (!batch) return;

    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i = 0;
    size_t limit = std::min(a.size(), b.size());
    while (i < limit && a[i] == b[i]) {
        i++;
    }
    return i;
}

bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t i = stop.size() - 1; i >= 0; --i)
        {
            if (stop[i] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, i + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - current_partial.length();
                }
            }
        }
    }
    return std::string::npos;
}

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    if (!ctx) return "<null_ctx>"; 
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (static_cast<unsigned int>(out[0]) & 0xff);
        std::string res(ss.str());
        if (res.length() == 1) {
            res = "0" + res;
        }
        out = "byte: \\x" + res;
    }
    return out;
}

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    if (!ctx) return "<null_ctx>"; 
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}

} // namespace cactus
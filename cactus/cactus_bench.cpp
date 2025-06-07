#include "cactus.h"
#include "llama.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace cactus {

std::string cactus_context::bench(int pp, int tg, int pl, int nr)
{
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("[]");
    }
    if (!ctx || !model) {
        LOG_ERROR("Context or model not initialized for benchmarking.");
        return std::string("[]");
    }

    is_predicting = true;

    double pp_avg = 0.0;
    double tg_avg = 0.0;
    double pp_std = 0.0;
    double tg_std = 0.0;

    int batch_size = std::min(pp, (int)params.n_batch);
    if (batch_size <= 0) {
         LOG_ERROR("Invalid batch size for benchmark: %d (pp=%d, n_batch=%d)", batch_size, pp, params.n_batch);
         is_predicting = false;
         return std::string("[]");
    }

    llama_batch batch = llama_batch_init(
        batch_size,
        0,
        pl
    );

    if (!batch.token) {
        LOG_ERROR("Failed to initialize llama_batch for benchmark.");
        is_predicting = false;
        return std::string("[]");
    }

    LOG_INFO("Starting benchmark: pp=%d, tg=%d, pl=%d, nr=%d, batch_size=%d", pp, tg, pl, nr, batch_size);

    for (int i = 0; i < nr; ++i)
    {
        if (is_interrupted) {
            LOG_INFO("Benchmark interrupted.");
            break;
        }

        llama_batch_clear(&batch);
        const int n_tokens_pp = pp;
        for (int k = 0; k < n_tokens_pp; ++k) {
             if (batch.n_tokens >= batch_size) {
                 LOG_ERROR("Benchmark batch capacity (%d) exceeded during PP phase.", batch_size);
                 goto cleanup_and_exit;
             }
             llama_batch_add(&batch, 0, k, {0}, false); 
        }
        if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = 1; 
        }

        llama_kv_self_clear(ctx);

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt processing benchmark", "");
             continue;
        }
        const int64_t t_pp_end = llama_time_us();

        if (is_interrupted) {
             LOG_INFO("Benchmark interrupted after PP phase.");
             break;
        }

        const int64_t t_tg_start = llama_time_us();
        int n_past_tg = batch.n_tokens;

        for (int k = 0; k < tg; ++k)
        {
            llama_batch_clear(&batch);
            for (int j = 0; j < pl; ++j)
            {
                 if (batch.n_tokens >= batch_size) {
                      LOG_ERROR("Benchmark batch capacity (%d) exceeded during TG phase.", batch_size);
                      goto cleanup_and_exit; 
                 }
                 llama_batch_add(&batch, 0, n_past_tg + k, {(llama_seq_id)j}, true); 
            }

            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation benchmark", "");
                 break; 
            }
            if (is_interrupted) {
                 LOG_INFO("Benchmark interrupted during TG phase.");
                 goto cleanup_and_exit;
            }
        }

        const int64_t t_tg_end = llama_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        const double speed_pp = (t_pp > 0) ? (double)n_tokens_pp / t_pp : 0.0;
        const double speed_tg = (t_tg > 0) ? (double)(pl * tg) / t_tg : 0.0; 

        pp_avg += speed_pp;
        tg_avg += speed_tg;
        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

cleanup_and_exit:
    llama_batch_free(batch);
    llama_kv_self_clear(ctx);
    is_predicting = false;

    int valid_repetitions = is_interrupted ? 0 : nr;
    if (valid_repetitions > 0) {
        pp_avg /= valid_repetitions;
        tg_avg /= valid_repetitions;

        if (valid_repetitions > 1) {
             double pp_var = pp_std / (valid_repetitions - 1) - pp_avg * pp_avg * valid_repetitions / (valid_repetitions - 1);
             double tg_var = tg_std / (valid_repetitions - 1) - tg_avg * tg_avg * valid_repetitions / (valid_repetitions - 1);
             pp_std = (pp_var > 0) ? sqrt(pp_var) : 0.0;
             tg_std = (tg_var > 0) ? sqrt(tg_var) : 0.0;
        } else {
            pp_std = 0.0;
            tg_std = 0.0;
        }
    } else {
         pp_avg = 0.0; tg_avg = 0.0; pp_std = 0.0; tg_std = 0.0;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    std::string result_str = "[\"" + std::string(model_desc) + "\"," +
                             std::to_string(llama_model_size(model)) + "," +
                             std::to_string(llama_model_n_params(model)) + "," +
                             std::to_string(pp_avg) + "," +
                             std::to_string(pp_std) + "," +
                             std::to_string(tg_avg) + "," +
                             std::to_string(tg_std) +
                             "]";
    LOG_INFO("Benchmark finished. Result: %s", result_str.c_str());
    return result_str;
}

} // namespace cactus
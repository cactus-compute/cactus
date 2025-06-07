#include "cactus.h"
#include "common.h"
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cmath>
#include <thread>

namespace cactus {

static const std::string default_audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";

static const std::string default_audio_data = R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>)";

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

static std::string process_text(const std::string & text, const tts_type tts_type = TTS_OUTETTS_V0_2) {
    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    std::string separator = (tts_type == TTS_OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env);

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

bool cactus_context::initVocoder(const std::string &vocoder_model_path) {
    if (vocoder_wrapper != nullptr) {
        return true;
    }
    
    common_params vocoder_params = params;
    vocoder_params.model.path = vocoder_model_path;
    vocoder_params.embedding = true;
    vocoder_params.n_ubatch = vocoder_params.n_batch;

    cactus_context_vocoder *wrapper = new cactus_context_vocoder{
        .init_result = common_init_from_params(vocoder_params),
    };

    wrapper->model = wrapper->init_result.model.get();
    wrapper->ctx = wrapper->init_result.context.get();

    if (wrapper->model == nullptr || wrapper->ctx == nullptr) {
        LOG_ERROR("Failed to load vocoder model: %s", vocoder_model_path.c_str());
        delete wrapper;
        return false;
    }

    wrapper->type = TTS_OUTETTS_V0_2;
    vocoder_wrapper = wrapper;
    has_vocoder = true;
    
    LOG_INFO("Vocoder initialized successfully with model: %s", vocoder_model_path.c_str());
    return true;
}

bool cactus_context::isVocoderEnabled() const {
    return has_vocoder && vocoder_wrapper != nullptr;
}

void cactus_context::releaseVocoder() {
    if (vocoder_wrapper != nullptr) {
        delete vocoder_wrapper;
        vocoder_wrapper = nullptr;
    }
    has_vocoder = false;
}

tts_type cactus_context::getTTSType() const {
    if (vocoder_wrapper == nullptr) {
        return TTS_UNKNOWN;
    }
    
    if (vocoder_wrapper->type != TTS_UNKNOWN) {
        return vocoder_wrapper->type;
    }
    
    if (model) {
        const char *chat_template = llama_model_chat_template(model, nullptr);
        if (chat_template && std::string(chat_template) == "outetts-0.3") {
            return TTS_OUTETTS_V0_3;
        }
    }
    return TTS_OUTETTS_V0_2;
}

std::string cactus_context::getFormattedAudioCompletion(const std::string &speaker_json_str, const std::string &text_to_speak) {
    if (!isVocoderEnabled()) {
        throw std::runtime_error("Vocoder is not enabled but audio completion is requested");
    }
    
    std::string audio_text = default_audio_text;
    std::string audio_data = default_audio_data;

    const tts_type type = getTTSType();
    if (type == TTS_UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return "";
    }

    if (type == TTS_OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    return "<|im_start|>\n" + audio_text + process_text(text_to_speak, type) + "<|text_end|>\n" + audio_data + "\n";
}

std::vector<llama_token> cactus_context::getAudioCompletionGuideTokens(const std::string &text_to_speak) {
    if (!model) {
        LOG_ERROR("Model not loaded for guide token generation");
        return {};
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const tts_type type = getTTSType();
    std::string clean_text = process_text(text_to_speak, type);

    const std::string& delimiter = (type == TTS_OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = clean_text.find(delimiter);

    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = clean_text.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        if (!tmp.empty()) {
            result.push_back(tmp[0]);
        }
        start = end + delimiter.length();
        end = clean_text.find(delimiter, start);
    }

    std::string current_word = clean_text.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (!tmp.empty()) {
        result.push_back(tmp[0]);
    }
    
    return result;
}

std::vector<float> cactus_context::decodeAudioTokens(const std::vector<llama_token> &tokens) {
    if (!isVocoderEnabled()) {
        throw std::runtime_error("Vocoder is not enabled but audio decoding is requested");
    }
    
    std::vector<llama_token> tokens_audio = tokens;
    tts_type type = getTTSType();
    
    if (type == TTS_OUTETTS_V0_3 || type == TTS_OUTETTS_V0_2) {
        tokens_audio.erase(std::remove_if(tokens_audio.begin(), tokens_audio.end(), 
            [](llama_token t) { return t < 151672 || t > 155772; }), tokens_audio.end());
        
        for (auto & token : tokens_audio) {
            token -= 151672;
        }
    } else {
        LOG_ERROR("Unsupported audio token type");
        return std::vector<float>();
    }
    
    const int n_codes = tokens_audio.size();
    if (n_codes == 0) {
        LOG_WARNING("No valid audio tokens found");
        return std::vector<float>();
    }
    
    llama_batch batch = llama_batch_init(n_codes, 0, 1);
    for (size_t i = 0; i < tokens_audio.size(); ++i) {
        llama_batch_add(&batch, tokens_audio[i], i, { 0 }, true);
    }
    
    if (batch.n_tokens != n_codes) {
        LOG_ERROR("batch.n_tokens != n_codes: %d != %d", batch.n_tokens, n_codes);
        llama_batch_free(batch);
        return std::vector<float>();
    }
    
    if (llama_encode(vocoder_wrapper->ctx, batch) != 0) {
        LOG_ERROR("llama_encode() failed");
        llama_batch_free(batch);
        return std::vector<float>();
    }
    
    llama_synchronize(vocoder_wrapper->ctx);
    const int n_embd = llama_model_n_embd(vocoder_wrapper->model);
    const float * embd = llama_get_embeddings(vocoder_wrapper->ctx);
    
    std::vector<float> audio_output = embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads);
    
    llama_batch_free(batch);
    return audio_output;
}

} // namespace cactus 
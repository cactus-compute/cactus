#import "CactusContext.h"
#import <Metal/Metal.h>

// Internal C++ headers - only for implementation, not exposed to framework consumers
#include <string>
#include <vector>
#include <iostream>
#include "cactus.h"
#include "llama.h"
#include "llama-impl.h"
#include "ggml.h"
#include "gguf.h"
#include "common.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"

@implementation CactusContext

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog {
  if (enabled) {
      void (^copiedBlock)(NSString *, NSString *) = [onEmitLog copy];
      llama_log_set([](lm_ggml_log_level level, const char * text, void * data) {
          llama_log_callback_default(level, text, data);
          NSString *levelStr = @"";
          if (level == LM_GGML_LOG_LEVEL_ERROR) {
              levelStr = @"error";
          } else if (level == LM_GGML_LOG_LEVEL_INFO) {
              levelStr = @"info";
          } else if (level == LM_GGML_LOG_LEVEL_WARN) {
              levelStr = @"warn";
          }

          NSString *textStr = [NSString stringWithUTF8String:text];
          // NOTE: Convert to UTF-8 string may fail
          if (!textStr) {
              return;
          }
          void (^block)(NSString *, NSString *) = (__bridge void (^)(NSString *, NSString *))(data);
          block(levelStr, textStr);
      }, copiedBlock);
  } else {
      llama_log_set(llama_log_callback_default, nullptr);
  }
}

+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip {
    struct lm_gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct lm_gguf_context * ctx = lm_gguf_init_from_file([path UTF8String], params);

    if (!ctx) {
        NSLog(@"%s: failed to load '%s'\n", __func__, [path UTF8String]);
        return @{};
    }

    NSMutableDictionary *info = [[NSMutableDictionary alloc] init];

    info[@"version"] = @(lm_gguf_get_version(ctx));
    info[@"alignment"] = @(lm_gguf_get_alignment(ctx));
    info[@"data_offset"] = @(lm_gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = lm_gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = lm_gguf_get_key(ctx, i);

            if (skip && [skip containsObject:[NSString stringWithUTF8String:key]]) {
                continue;
            }
            const std::string value = lm_gguf_kv_to_str(ctx, i);
            info[[NSString stringWithUTF8String:key]] = [NSString stringWithUTF8String:value.c_str()];
        }
    }

    lm_gguf_free(ctx);

    return info;
}

+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress {
    // llama_backend_init(false);
    common_params defaultParams;

    if (params[@"vocab_only"]) {
        defaultParams.vocab_only = [params[@"vocab_only"] boolValue];
        defaultParams.warmup = false;
    }

    NSString *modelPath = params[@"model"];
    BOOL isAsset = [params[@"is_model_asset"] boolValue];
    NSString *path = modelPath;
    if (isAsset) path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    defaultParams.model = [path UTF8String];

    NSString *chatTemplate = params[@"chat_template"];
    if (chatTemplate) {
        defaultParams.chat_template = [chatTemplate UTF8String];
        NSLog(@"chatTemplate: %@", chatTemplate);
    }

    NSString *reasoningFormat = params[@"reasoning_format"];
    if (reasoningFormat && [reasoningFormat isEqualToString:@"deepseek"]) {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    } else {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    }

    if (params[@"n_ctx"]) defaultParams.n_ctx = [params[@"n_ctx"] intValue];
    if (params[@"use_mlock"]) defaultParams.use_mlock = [params[@"use_mlock"]boolValue];

    BOOL skipGpuDevices = params[@"no_gpu_devices"] && [params[@"no_gpu_devices"] boolValue];

    BOOL isMetalEnabled = false;
    NSString *reasonNoMetal = @"";
    defaultParams.n_gpu_layers = 0;
#ifdef LM_GGML_USE_METAL
    // Check ggml-metal availability
    NSError * error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device
        newLibraryWithSource:@"#include <metal_stdlib>\n"
                                "using namespace metal;"
                                "typedef matrix<bfloat, 4, 4> bfloat4x4;"
                                "kernel void test() { simd_sum(0); }"
        options:nil
        error:&error
    ];
    if (error) {
        reasonNoMetal = [error localizedDescription];
        skipGpuDevices = true;
    } else {
        id<MTLFunction> kernel = [library newFunctionWithName:@"test"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
        if (pipeline == nil) {
            reasonNoMetal = [error localizedDescription];
            skipGpuDevices = true;
        } else {
#if TARGET_OS_SIMULATOR
            // Use the backend, but no layers because not supported fully on simulator
            defaultParams.n_gpu_layers = 0;
            isMetalEnabled = true;
#else
            defaultParams.n_gpu_layers = [params[@"n_gpu_layers"] intValue];
            isMetalEnabled = true;
#endif
        }
    }
    device = nil;
#else
    reasonNoMetal = @"Metal is not enabled in this build";
    isMetalEnabled = false;
#endif

    if (skipGpuDevices) {
        std::vector<lm_ggml_backend_dev_t> cpu_devs;
        for (size_t i = 0; i < lm_ggml_backend_dev_count(); ++i) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
            switch (lm_ggml_backend_dev_type(dev)) {
                case LM_GGML_BACKEND_DEVICE_TYPE_CPU:
                case LM_GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    cpu_devs.push_back(dev);
                    break;
                case LM_GGML_BACKEND_DEVICE_TYPE_GPU:
                    break;
            }
        }
        if (cpu_devs.size() > 0) {
            defaultParams.devices = cpu_devs;
        }
    }

    if (params[@"n_batch"]) defaultParams.n_batch = [params[@"n_batch"] intValue];
    if (params[@"n_ubatch"]) defaultParams.n_ubatch = [params[@"n_ubatch"] intValue];
    if (params[@"use_mmap"]) defaultParams.use_mmap = [params[@"use_mmap"] boolValue];

    if (params[@"pooling_type"] && [params[@"pooling_type"] isKindOfClass:[NSNumber class]]) {
      defaultParams.pooling_type = static_cast<enum llama_pooling_type>([params[@"pooling_type"] intValue]);
    }

    if (params[@"embedding"] && [params[@"embedding"] boolValue]) {
        defaultParams.embedding = true;
        // For non-causal models, batch size must be equal to ubatch size
        defaultParams.n_ubatch = defaultParams.n_batch;

        if (params[@"embd_normalize"] && [params[@"embd_normalize"] isKindOfClass:[NSNumber class]]) {
            defaultParams.embd_normalize = [params[@"embd_normalize"] intValue];
        }
    }

    if (params[@"rope_freq_base"]) defaultParams.rope_freq_base = [params[@"rope_freq_base"] floatValue];
    if (params[@"rope_freq_scale"]) defaultParams.rope_freq_scale = [params[@"rope_freq_scale"] floatValue];

    if (params[@"flash_attn"] && [params[@"flash_attn"] boolValue]) defaultParams.flash_attn = true;

    if (params[@"cache_type_k"]) defaultParams.cache_type_k = cactus::kv_cache_type_from_str([params[@"cache_type_k"] UTF8String]);
    if (params[@"cache_type_v"]) defaultParams.cache_type_v = cactus::kv_cache_type_from_str([params[@"cache_type_v"] UTF8String]);

    int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : 0;
    const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
    defaultParams.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;

    CactusContext *context = [[CactusContext alloc] init];
    context->llama = new cactus::cactus_context();
    context->llama->is_load_interrupted = false;
    context->llama->loading_progress = 0;
    context->onProgress = onProgress;

    if (params[@"use_progress_callback"] && [params[@"use_progress_callback"] boolValue]) {
        defaultParams.progress_callback = [](float progress, void * user_data) {
            CactusContext *context = (__bridge CactusContext *)(user_data);
            unsigned percentage = (unsigned) (100 * progress);
            if (percentage > context->llama->loading_progress) {
                context->llama->loading_progress = percentage;
                context->onProgress(percentage);
            }
            return !context->llama->is_load_interrupted;
        };
        defaultParams.progress_callback_user_data = context;
    }

    context->is_model_loaded = context->llama->loadModel(defaultParams);

    if (
        params[@"embedding"] && [params[@"embedding"] boolValue] &&
        llama_model_has_encoder(context->llama->model) && llama_model_has_decoder(context->llama->model)
    ) {
        delete context->llama;
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not supported in encoder-decoder models" userInfo:nil];
    }

    std::vector<common_adapter_lora_info> lora;
    if (params[@"lora"]) {
        common_adapter_lora_info la;
        la.path = [params[@"lora"] UTF8String];
        la.scale = 1.0f;
        if (params[@"lora_scaled"]) la.scale = [params[@"lora_scaled"] floatValue];
        lora.push_back(la);
    }
    if (params[@"lora_list"] && [params[@"lora_list"] isKindOfClass:[NSArray class]]) {
        NSArray *lora_list = params[@"lora_list"];
        for (NSDictionary *lora_adapter in lora_list) {
          NSString *path = lora_adapter[@"path"];
          if (!path) continue;
          float scale = [lora_adapter[@"scaled"] floatValue];
          common_adapter_lora_info la;
          la.path = [path UTF8String];
          la.scale = scale;
          lora.push_back(la);
        }
    }
    if (lora.size() > 0) {
        int result = context->llama->applyLoraAdapters(lora);
        if (result != 0) {
            delete context->llama;
            @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapters" userInfo:nil];
        }
    }

    context->is_metal_enabled = isMetalEnabled;
    context->reason_no_metal = reasonNoMetal;

    return context;
}

- (void)interruptLoad {
    llama->is_load_interrupted = true;
}

- (bool)isMetalEnabled {
    return is_metal_enabled;
}

- (NSString *)reasonNoMetal {
    return reason_no_metal;
}

- (NSDictionary *)modelInfo {
    char desc[1024];
    llama_model_desc(llama->model, desc, sizeof(desc));

    int count = llama_model_meta_count(llama->model);
    NSDictionary *meta = [[NSMutableDictionary alloc] init];
    for (int i = 0; i < count; i++) {
        char key[256];
        llama_model_meta_key_by_index(llama->model, i, key, sizeof(key));
        char val[4096];
        llama_model_meta_val_str_by_index(llama->model, i, val, sizeof(val));

        NSString *keyStr = [NSString stringWithUTF8String:key];
        NSString *valStr = [NSString stringWithUTF8String:val];
        [meta setValue:valStr forKey:keyStr];
    }

    auto template_tool_use = llama->templates.get()->template_tool_use.get();
    NSDictionary *tool_use_caps_dir = nil;
    if (template_tool_use) {
        auto tool_use_caps = template_tool_use->original_caps();
        tool_use_caps_dir = @{
            @"tools": @(tool_use_caps.supports_tools),
            @"toolCalls": @(tool_use_caps.supports_tool_calls),
            @"toolResponses": @(tool_use_caps.supports_tool_responses),
            @"systemRole": @(tool_use_caps.supports_system_role),
            @"parallelToolCalls": @(tool_use_caps.supports_parallel_tool_calls),
            @"toolCallId": @(tool_use_caps.supports_tool_call_id)
        };
    }

    auto default_tmpl = llama->templates.get()->template_default.get();
    auto default_tmpl_caps = default_tmpl->original_caps();

    return @{
        @"desc": [NSString stringWithUTF8String:desc],
        @"size": @(llama_model_size(llama->model)),
        @"nEmbd": @(llama_model_n_embd(llama->model)),
        @"nParams": @(llama_model_n_params(llama->model)),
        @"chatTemplates": @{
            @"llamaChat": @(llama->validateModelChatTemplate(false, nullptr)),
            @"minja": @{
                @"default": @(llama->validateModelChatTemplate(true, nullptr)),
                @"defaultCaps": @{
                    @"tools": @(default_tmpl_caps.supports_tools),
                    @"toolCalls": @(default_tmpl_caps.supports_tool_calls),
                    @"toolResponses": @(default_tmpl_caps.supports_tool_responses),
                    @"systemRole": @(default_tmpl_caps.supports_system_role),
                    @"parallelToolCalls": @(default_tmpl_caps.supports_parallel_tool_calls),
                    @"toolCallId": @(default_tmpl_caps.supports_tool_call_id)
                },
                @"toolUse": @(llama->validateModelChatTemplate(true, "tool_use")),
                @"toolUseCaps": tool_use_caps_dir ?: @{}
            }
        },
        @"metadata": meta,

        // deprecated
        @"isChatTemplateSupported": @(llama->validateModelChatTemplate(false, nullptr))
    };
}

- (bool)isModelLoaded {
    return is_model_loaded;
}

- (bool)isPredicting {
    return llama->is_predicting;
}

- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
    withChatTemplate:(NSString *)chatTemplate
    withJsonSchema:(NSString *)jsonSchema
    withTools:(NSString *)tools
    withParallelToolCalls:(BOOL)parallelToolCalls
    withToolChoice:(NSString *)toolChoice
{
    auto tmpl_str = chatTemplate == nil ? "" : [chatTemplate UTF8String];

    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    auto chatParams = llama->getFormattedChatWithJinja(
        [messages UTF8String],
        tmpl_str,
        jsonSchema == nil ? "" : [jsonSchema UTF8String],
        tools == nil ? "" : [tools UTF8String],
        parallelToolCalls,
        toolChoice == nil ? "" : [toolChoice UTF8String]
    );
    result[@"prompt"] = [NSString stringWithUTF8String:chatParams.prompt.c_str()];
    result[@"chat_format"] = @(static_cast<int>(chatParams.format));
    result[@"grammar"] = [NSString stringWithUTF8String:chatParams.grammar.c_str()];
    result[@"grammar_lazy"] = @(chatParams.grammar_lazy);
    NSMutableArray *grammar_triggers = [[NSMutableArray alloc] init];
    for (const auto & trigger : chatParams.grammar_triggers) {
        [grammar_triggers addObject:@{
            @"type": @(trigger.type),
            @"value": [NSString stringWithUTF8String:trigger.value.c_str()],
            @"token": @(trigger.token),
        }];
    }
    result[@"grammar_triggers"] = grammar_triggers;
    NSMutableArray *preserved_tokens = [[NSMutableArray alloc] init];
    for (const auto & token : chatParams.preserved_tokens) {
        [preserved_tokens addObject:[NSString stringWithUTF8String:token.c_str()]];
    }
    result[@"preserved_tokens"] = preserved_tokens;
    NSMutableArray *additional_stops = [[NSMutableArray alloc] init];
    for (const auto & stop : chatParams.additional_stops) {
        [additional_stops addObject:[NSString stringWithUTF8String:stop.c_str()]];
    }
    result[@"additional_stops"] = additional_stops;

    return result;
}

- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate {
    auto tmpl_str = chatTemplate == nil ? "" : [chatTemplate UTF8String];
    return [NSString stringWithUTF8String:llama->getFormattedChat(
        [messages UTF8String],
        tmpl_str
    ).c_str()];;
}

- (NSArray *)tokenProbsToDict:(std::vector<cactus::completion_token_output>)probs {
    NSMutableArray *out = [[NSMutableArray alloc] init];
    for (const auto &prob : probs)
    {
        NSMutableArray *probsForToken = [[NSMutableArray alloc] init];
        for (const auto &p : prob.probs)
        {
            std::string tokStr = cactus::tokens_to_output_formatted_string(llama->ctx, p.tok);
            [probsForToken addObject:@{
                @"tok_str": [NSString stringWithUTF8String:tokStr.c_str()],
                @"prob": [NSNumber numberWithDouble:p.prob]
            }];
        }
        std::string tokStr = cactus::tokens_to_output_formatted_string(llama->ctx, prob.tok);
        [out addObject:@{
            @"content": [NSString stringWithUTF8String:tokStr.c_str()],
            @"probs": probsForToken
        }];
    }
    return out;
}

- (NSDictionary *)completion:(NSDictionary *)params
    onToken:(void (^)(NSMutableDictionary * tokenResult))onToken
{
    llama->rewind();

    //llama_reset_timings(llama->ctx);

    NSString *prompt = [params objectForKey:@"prompt"];

    llama->params.prompt = [prompt UTF8String];
    llama->params.sampling.seed = params[@"seed"] ? [params[@"seed"] intValue] : -1;

    if (params[@"n_threads"]) {
        int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : llama->params.cpuparams.n_threads;
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        // Use 2 threads by default on 4-core devices, 4 threads on more cores
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        llama->params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) llama->params.n_predict = [params[@"n_predict"] intValue];
    if (params[@"ignore_eos"]) llama->params.sampling.ignore_eos = [params[@"ignore_eos"] boolValue];

    auto & sparams = llama->params.sampling;

    if (params[@"temperature"]) sparams.temp = [params[@"temperature"] doubleValue];

    if (params[@"n_probs"]) sparams.n_probs = [params[@"n_probs"] intValue];

    if (params[@"penalty_last_n"]) sparams.penalty_last_n = [params[@"penalty_last_n"] intValue];
    if (params[@"penalty_repeat"]) sparams.penalty_repeat = [params[@"penalty_repeat"] doubleValue];
    if (params[@"penalty_freq"]) sparams.penalty_freq = [params[@"penalty_freq"] doubleValue];
    if (params[@"penalty_present"]) sparams.penalty_present = [params[@"penalty_present"] doubleValue];

    if (params[@"mirostat"]) sparams.mirostat = [params[@"mirostat"] intValue];
    if (params[@"mirostat_tau"]) sparams.mirostat_tau = [params[@"mirostat_tau"] doubleValue];
    if (params[@"mirostat_eta"]) sparams.mirostat_eta = [params[@"mirostat_eta"] doubleValue];

    if (params[@"top_k"]) sparams.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) sparams.top_p = [params[@"top_p"] doubleValue];
    if (params[@"min_p"]) sparams.min_p = [params[@"min_p"] doubleValue];
    if (params[@"xtc_threshold"]) sparams.xtc_threshold = [params[@"xtc_threshold"] doubleValue];
    if (params[@"xtc_probability"]) sparams.xtc_probability = [params[@"xtc_probability"] doubleValue];
    if (params[@"typical_p"]) sparams.typ_p = [params[@"typical_p"] doubleValue];

    if (params[@"dry_multiplier"]) sparams.dry_multiplier = [params[@"dry_multiplier"] doubleValue];
    if (params[@"dry_base"]) sparams.dry_base = [params[@"dry_base"] doubleValue];
    if (params[@"dry_allowed_length"]) sparams.dry_allowed_length = [params[@"dry_allowed_length"] intValue];
    if (params[@"dry_penalty_last_n"]) sparams.dry_penalty_last_n = [params[@"dry_penalty_last_n"] intValue];

    if (params[@"top_n_sigma"]) sparams.top_n_sigma = [params[@"top_n_sigma"] doubleValue];

    // dry break seq
    if (params[@"dry_sequence_breakers"] && [params[@"dry_sequence_breakers"] isKindOfClass:[NSArray class]]) {
        NSArray *dry_sequence_breakers = params[@"dry_sequence_breakers"];
        for (NSString *s in dry_sequence_breakers) {
            sparams.dry_sequence_breakers.push_back([s UTF8String]);
        }
    }

    if (params[@"grammar"]) {
        sparams.grammar = [params[@"grammar"] UTF8String];
    }

    if (params[@"json_schema"] && !params[@"grammar"]) {
        sparams.grammar = json_schema_to_grammar(json::parse([params[@"json_schema"] UTF8String]));
    }

    if (params[@"grammar_lazy"]) {
        sparams.grammar_lazy = [params[@"grammar_lazy"] boolValue];
    }

    if (params[@"preserved_tokens"] && [params[@"preserved_tokens"] isKindOfClass:[NSArray class]]) {
        NSArray *preserved_tokens = params[@"preserved_tokens"];
        for (NSString *token in preserved_tokens) {
            auto ids = common_tokenize(llama->ctx, [token UTF8String], /* add_special= */ false, /* parse_special= */ true);
            if (ids.size() == 1) {
                sparams.preserved_tokens.insert(ids[0]);
            } else {
//                LOG_WRN("Not preserved because more than 1 token (wrong chat template override?): %s\n", [token UTF8String]);
            }
        }
    }

    if (params[@"grammar_triggers"] && [params[@"grammar_triggers"] isKindOfClass:[NSArray class]]) {
        NSArray *grammar_triggers = params[@"grammar_triggers"];
        for (NSDictionary *grammar_trigger in grammar_triggers) {
            const auto type = static_cast<common_grammar_trigger_type>([grammar_trigger[@"type"] intValue]);
            const auto & word = [grammar_trigger[@"value"] UTF8String];

            if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
              auto ids = common_tokenize(llama->ctx, word, /* add_special= */ false, /* parse_special= */ true);
              if (ids.size() == 1) {
                  auto token = ids[0];
                  if (std::find(sparams.preserved_tokens.begin(), sparams.preserved_tokens.end(), (llama_token) token) == sparams.preserved_tokens.end()) {
                      throw std::runtime_error("Grammar trigger word should be marked as preserved token");
                  }
                  common_grammar_trigger trigger;
                  trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                  trigger.value = word;
                  trigger.token = token;
                  sparams.grammar_triggers.push_back(std::move(trigger));
              } else {
                  sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
              }
            } else {
                common_grammar_trigger trigger;
                trigger.type = type;
                trigger.value = word;
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    const auto token = (llama_token) [grammar_trigger[@"token"] intValue];
                    trigger.token = token;
                }
                sparams.grammar_triggers.push_back(std::move(trigger));
            }
        }
    }

    llama->params.antiprompt.clear();
    if (params[@"stop"]) {
        NSArray *stop = params[@"stop"];
        for (NSString *s in stop) {
            llama->params.antiprompt.push_back([s UTF8String]);
        }
    }

    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    sparams.logit_bias.clear();
    if (params[@"ignore_eos"] && [params[@"ignore_eos"] boolValue]) {
        sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
    }

    if (params[@"logit_bias"] && [params[@"logit_bias"] isKindOfClass:[NSArray class]]) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        NSArray *logit_bias = params[@"logit_bias"];
        for (NSArray *el in logit_bias) {
            if ([el isKindOfClass:[NSArray class]] && [el count] == 2) {
                llama_token tok = [el[0] intValue];
                if (tok >= 0 && tok < n_vocab) {
                    if ([el[1] isKindOfClass:[NSNumber class]]) {
                        sparams.logit_bias[tok].bias = [el[1] doubleValue];
                    } else if ([el[1] isKindOfClass:[NSNumber class]] && ![el[1] boolValue]) {
                        sparams.logit_bias[tok].bias = -INFINITY;
                    }
                }
            }
        }
    }

    if (!llama->initSampling()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to initialize sampling" userInfo:nil];
    }
    llama->beginCompletion();
    llama->loadPrompt();

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->has_next_token && !llama->is_interrupted) {
        const cactus::completion_token_output token_with_probs = llama->doCompletion();
        if (token_with_probs.tok == -1 || llama->incomplete) {
            continue;
        }
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->generated_text.size());

        const std::string str_test = llama->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->findStoppingStrings(str_test, token_text.size(), cactus::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->generated_text.erase(
                llama->generated_text.begin() + pos + stop_pos,
                llama->generated_text.end());
            pos = std::min(sent_count, llama->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->findStoppingStrings(str_test, token_text.size(),
                cactus::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<cactus::completion_token_output> probs_output = {};

            NSMutableDictionary *tokenResult = [[NSMutableDictionary alloc] init];
            tokenResult[@"token"] = [NSString stringWithUTF8String:to_send.c_str()];

            if (llama->params.sampling.n_probs > 0) {
                const std::vector<llama_token> to_send_toks = common_tokenize(llama->ctx, to_send, false);
                size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
                size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
                if (probs_pos < probs_stop_pos) {
                    probs_output = std::vector<cactus::completion_token_output>(llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
                }
                sent_token_probs_index = probs_stop_pos;

                tokenResult[@"completion_probabilities"] = [self tokenProbsToDict:probs_output];
            }

            onToken(tokenResult);
        }
    }

    llama_perf_context_print(llama->ctx);
    llama->is_predicting = false;

    const auto timings = llama_perf_context(llama->ctx);

    NSMutableArray *toolCalls = nil;
    NSString *reasoningContent = nil;
    NSString *content = nil;
    if (!llama->is_interrupted) {
        try {
            auto chat_format = params[@"chat_format"] ? [params[@"chat_format"] intValue] : COMMON_CHAT_FORMAT_CONTENT_ONLY;
            common_chat_msg message = common_chat_parse(llama->generated_text, static_cast<common_chat_format>(chat_format));
            if (!message.reasoning_content.empty()) {
                reasoningContent = [NSString stringWithUTF8String:message.reasoning_content.c_str()];
            }
            content = [NSString stringWithUTF8String:message.content.c_str()];
            toolCalls = [[NSMutableArray alloc] init];
            for (const auto &tc : message.tool_calls) {
                [toolCalls addObject:@{
                    @"type": @"function",
                    @"function": @{
                        @"name": [NSString stringWithUTF8String:tc.name.c_str()],
                        @"arguments": [NSString stringWithUTF8String:tc.arguments.c_str()],
                    },
                    @"id": tc.id.empty() ? [NSNull null] : [NSString stringWithUTF8String:tc.id.c_str()],
                }];
            }
        } catch (const std::exception &e) {
            // NSLog(@"Error parsing tool calls: %s", e.what());
        }
    }

    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    result[@"text"] = [NSString stringWithUTF8String:llama->generated_text.c_str()]; // Original text
    if (content) result[@"content"] = content;
    if (reasoningContent) result[@"reasoning_content"] = reasoningContent;
    if (toolCalls && toolCalls.count > 0) result[@"tool_calls"] = toolCalls;
    result[@"completion_probabilities"] = [self tokenProbsToDict:llama->generated_token_probs];
    result[@"tokens_predicted"] = @(llama->num_tokens_predicted);
    result[@"tokens_evaluated"] = @(llama->num_prompt_tokens);
    result[@"truncated"] = @(llama->truncated);
    result[@"stopped_eos"] = @(llama->stopped_eos);
    result[@"stopped_word"] = @(llama->stopped_word);
    result[@"stopped_limit"] = @(llama->stopped_limit);
    result[@"stopping_word"] = [NSString stringWithUTF8String:llama->stopping_word.c_str()];
    result[@"tokens_cached"] = @(llama->n_past);
    result[@"timings"] = @{
        @"prompt_n": @(timings.n_p_eval),
        @"prompt_ms": @(timings.t_p_eval_ms),
        @"prompt_per_token_ms": @(timings.t_p_eval_ms / timings.n_p_eval),
        @"prompt_per_second": @(1e3 / timings.t_p_eval_ms * timings.n_p_eval),
        @"predicted_n": @(timings.n_eval),
        @"predicted_n": @(timings.n_eval),
        @"predicted_ms": @(timings.t_eval_ms),
        @"predicted_per_token_ms": @(timings.t_eval_ms / timings.n_eval),
        @"predicted_per_second": @(1e3 / timings.t_eval_ms * timings.n_eval),
    };
    return result;
}

- (void)stopCompletion {
    llama->is_interrupted = true;
}

- (NSArray *)tokenize:(NSString *)text {
    const std::vector<llama_token> toks = common_tokenize(llama->ctx, [text UTF8String], false);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (llama_token tok : toks) {
        [result addObject:@(tok)];
    }
    return result;
}

- (NSArray *)tokenize:(NSString *)text withMediaPaths:(NSArray *)mediaPaths {
    std::vector<std::string> media_paths_vector;
    if (mediaPaths) {
        for (NSString *mediaPath in mediaPaths) {
            media_paths_vector.push_back([mediaPath UTF8String]);
        }
    }
    
    cactus::cactus_tokenize_result tokenize_result = llama->tokenize([text UTF8String], media_paths_vector);
    
    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    
    // Add tokens array
    NSMutableArray *tokens = [[NSMutableArray alloc] init];
    for (const auto &tok : tokenize_result.tokens) {
        [tokens addObject:@(tok)];
    }
    result[@"tokens"] = tokens;
    
    // Add media info if present
    result[@"has_media"] = @(tokenize_result.has_media);
    
    if (tokenize_result.has_media) {
        NSMutableArray *bitmap_hashes = [[NSMutableArray alloc] init];
        for (const auto &hash : tokenize_result.bitmap_hashes) {
            [bitmap_hashes addObject:[NSString stringWithUTF8String:hash.c_str()]];
        }
        result[@"bitmap_hashes"] = bitmap_hashes;
        
        NSMutableArray *chunk_pos = [[NSMutableArray alloc] init];
        for (const auto &pos : tokenize_result.chunk_pos) {
            [chunk_pos addObject:@(pos)];
        }
        result[@"chunk_pos"] = chunk_pos;
        
        NSMutableArray *chunk_pos_media = [[NSMutableArray alloc] init];
        for (const auto &pos : tokenize_result.chunk_pos_media) {
            [chunk_pos_media addObject:@(pos)];
        }
        result[@"chunk_pos_media"] = chunk_pos_media;
    }
    
    return result;
}

- (NSString *)detokenize:(NSArray *)tokens {
    std::vector<llama_token> toks;
    for (NSNumber *tok in tokens) {
        toks.push_back([tok intValue]);
    }
    const std::string text = cactus::tokens_to_str(llama->ctx, toks.cbegin(), toks.cend());
    return [NSString stringWithUTF8String:text.c_str()];
}

- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params {
    if (llama->params.embedding != true) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not enabled" userInfo:nil];
    }

    common_params embdParams;
    embdParams.embedding = true;
    embdParams.embd_normalize = llama->params.embd_normalize;

    if (params[@"embd_normalize"] && [params[@"embd_normalize"] isKindOfClass:[NSNumber class]]) {
        embdParams.embd_normalize = [params[@"embd_normalize"] intValue];
    }

    llama->rewind();

    llama_perf_context_reset(llama->ctx);

    llama->params.prompt = [text UTF8String];

    llama->params.n_predict = 0;

    if (!llama->initSampling()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to initialize sampling" userInfo:nil];
    }
    llama->beginCompletion();
    llama->loadPrompt();
    llama->doCompletion();

    std::vector<float> result = llama->getEmbedding(embdParams);

    NSMutableDictionary *resultDict = [[NSMutableDictionary alloc] init];
    NSMutableArray *embeddingResult = [[NSMutableArray alloc] init];
    for (float f : result) {
        [embeddingResult addObject:@(f)];
    }
    resultDict[@"embedding"] = embeddingResult;
    NSMutableArray *promptTokens = [[NSMutableArray alloc] init];
    for (llama_token tok : llama->embd) {
        [promptTokens addObject:[NSString stringWithUTF8String:common_token_to_piece(llama->ctx, tok).c_str()]];
    }
    resultDict[@"prompt_tokens"] = promptTokens;

    llama->is_predicting = false;
    return resultDict;
}

- (NSDictionary *)loadSession:(NSString *)path {
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session file does not exist" userInfo:nil];
    }

    size_t n_token_count_out = 0;
    llama->embd.resize(llama->params.n_ctx);
    if (!llama_state_load_file(llama->ctx, [path UTF8String], llama->embd.data(), llama->embd.capacity(), &n_token_count_out)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to load session" userInfo:nil];
    }
    llama->embd.resize(n_token_count_out);
    const std::string text = cactus::tokens_to_str(llama->ctx, llama->embd.cbegin(), llama->embd.cend());
    return @{
        @"tokens_loaded": @(n_token_count_out),
        @"prompt": [NSString stringWithUTF8String:text.c_str()]
    };
}

- (int)saveSession:(NSString *)path size:(int)size {
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    std::vector<llama_token> session_tokens = llama->embd;
    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_state_save_file(llama->ctx, [path UTF8String], session_tokens.data(), save_size)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to save session" userInfo:nil];
    }
    return session_tokens.size();
}

- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr {
    return [NSString stringWithUTF8String:llama->bench(pp, tg, pl, nr).c_str()];
}

- (void)applyLoraAdapters:(NSArray *)loraAdapters {
    std::vector<common_adapter_lora_info> lora_adapters;
    for (NSDictionary *loraAdapter in loraAdapters) {
        common_adapter_lora_info la;
        la.path = [loraAdapter[@"path"] UTF8String];
        la.scale = [loraAdapter[@"scaled"] doubleValue];
        la.ptr = llama_adapter_lora_init(llama->model, la.path.c_str());
        if (la.ptr == nullptr) {
            @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapter" userInfo:nil];
        }
        lora_adapters.push_back(la);
    }
    int result = llama->applyLoraAdapters(lora_adapters);
    if (result != 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapters" userInfo:nil];
    }
}

- (void)removeLoraAdapters {
    llama->removeLoraAdapters();
}

- (NSArray *)getLoadedLoraAdapters {
    std::vector<common_adapter_lora_info> loaded_lora_adapters = llama->getLoadedLoraAdapters();
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (common_adapter_lora_info &la : loaded_lora_adapters) {
        [result addObject:@{
            @"path": [NSString stringWithUTF8String:la.path.c_str()],
            @"scale": @(la.scale)
        }];
    }
    return result;
}

// New Multimodal Methods
- (BOOL)initMultimodal:(NSString *)mmprojPath useGpu:(BOOL)useGpu {
    return llama->initMultimodal([mmprojPath UTF8String], useGpu);
}

- (BOOL)isMultimodalEnabled {
    return llama->isMultimodalEnabled();
}

- (BOOL)isMultimodalSupportVision {
    return llama->isMultimodalSupportVision();
}

- (BOOL)isMultimodalSupportAudio {
    return llama->isMultimodalSupportAudio();
}

- (void)releaseMultimodal {
    llama->releaseMultimodal();
}

- (NSDictionary *)multimodalCompletion:(NSString *)prompt 
    withMediaPaths:(NSArray *)mediaPaths 
    params:(NSDictionary *)params 
    onToken:(void (^)(NSMutableDictionary *tokenResult))onToken {
    
    if (!llama->isMultimodalEnabled()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Multimodal is not enabled" userInfo:nil];
    }
    
    llama->rewind();
    
    llama->params.prompt = [prompt UTF8String];
    llama->params.sampling.seed = params[@"seed"] ? [params[@"seed"] intValue] : -1;
    
    // Set all completion parameters (similar to existing completion method)
    if (params[@"n_threads"]) {
        int nThreads = [params[@"n_threads"] intValue];
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        llama->params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) llama->params.n_predict = [params[@"n_predict"] intValue];
    if (params[@"ignore_eos"]) llama->params.sampling.ignore_eos = [params[@"ignore_eos"] boolValue];
    
    auto & sparams = llama->params.sampling;
    if (params[@"temperature"]) sparams.temp = [params[@"temperature"] doubleValue];
    if (params[@"top_k"]) sparams.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) sparams.top_p = [params[@"top_p"] doubleValue];
    
    // Convert media paths
    std::vector<std::string> media_paths_vector;
    if (mediaPaths) {
        for (NSString *mediaPath in mediaPaths) {
            media_paths_vector.push_back([mediaPath UTF8String]);
        }
    }
    
    if (!llama->initSampling()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to initialize sampling" userInfo:nil];
    }
    
    llama->beginCompletion();
    
    @try {
        llama->loadPrompt(media_paths_vector);
    } @catch (NSException *exception) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:exception.reason userInfo:nil];
    }
    
    size_t sent_count = 0;
    while (llama->has_next_token && !llama->is_interrupted) {
        const cactus::completion_token_output token_with_probs = llama->doCompletion();
        if (token_with_probs.tok == -1 || llama->incomplete) {
            continue;
        }
        
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);
        size_t pos = std::min(sent_count, llama->generated_text.size());
        const std::string str_test = llama->generated_text.substr(pos);
        
        size_t stop_pos = llama->findStoppingStrings(str_test, token_text.size(), cactus::STOP_FULL);
        
        if (stop_pos == std::string::npos || (!llama->has_next_token && stop_pos > 0)) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);
            sent_count += to_send.size();
            
            if (onToken) {
                NSMutableDictionary *tokenResult = [[NSMutableDictionary alloc] init];
                tokenResult[@"token"] = [NSString stringWithUTF8String:to_send.c_str()];
                onToken(tokenResult);
            }
        }
    }
    
    llama->is_predicting = false;
    
    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    result[@"text"] = [NSString stringWithUTF8String:llama->generated_text.c_str()];
    result[@"tokens_predicted"] = @(llama->num_tokens_predicted);
    result[@"tokens_evaluated"] = @(llama->num_prompt_tokens);
    result[@"truncated"] = @(llama->truncated);
    result[@"stopped_eos"] = @(llama->stopped_eos);
    result[@"stopped_word"] = @(llama->stopped_word);
    result[@"stopped_limit"] = @(llama->stopped_limit);
    result[@"stopping_word"] = [NSString stringWithUTF8String:llama->stopping_word.c_str()];
    result[@"tokens_cached"] = @(llama->n_past);
    
    return result;
}

- (void)releaseVocoder {
    llama->releaseVocoder();
}

- (void)invalidate {
    delete llama;
    // llama_backend_free();
}

// New TTS/Vocoder Methods
- (BOOL)initVocoder:(NSString *)vocoderModelPath {
    return llama->initVocoder([vocoderModelPath UTF8String]);
}

- (BOOL)isVocoderEnabled {
    return llama->isVocoderEnabled();
}

- (int)getTTSType {
    return static_cast<int>(llama->getTTSType());
}

- (NSString *)getFormattedAudioCompletion:(NSString *)speakerJsonStr textToSpeak:(NSString *)textToSpeak {
    std::string result = llama->getFormattedAudioCompletion([speakerJsonStr UTF8String], [textToSpeak UTF8String]);
    return [NSString stringWithUTF8String:result.c_str()];
}

- (NSArray *)getAudioCompletionGuideTokens:(NSString *)textToSpeak {
    std::vector<llama_token> tokens = llama->getAudioCompletionGuideTokens([textToSpeak UTF8String]);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (const auto &token : tokens) {
        [result addObject:@(token)];
    }
    return result;
}

- (NSArray *)decodeAudioTokens:(NSArray *)tokens {
    std::vector<llama_token> token_vector;
    for (NSNumber *token in tokens) {
        token_vector.push_back([token intValue]);
    }
    
    std::vector<float> audio_data = llama->decodeAudioTokens(token_vector);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (const auto &sample : audio_data) {
        [result addObject:@(sample)];
    }
    return result;
}

@end

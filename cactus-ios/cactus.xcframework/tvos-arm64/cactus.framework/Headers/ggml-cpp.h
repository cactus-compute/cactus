#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <memory>

// Smart pointers for ggml types

// ggml

struct lm_ggml_context_deleter { void operator()(lm_ggml_context * ctx) { lm_ggml_free(ctx); } };
struct lm_gguf_context_deleter { void operator()(lm_gguf_context * ctx) { lm_gguf_free(ctx); } };

typedef std::unique_ptr<lm_ggml_context, lm_ggml_context_deleter> lm_ggml_context_ptr;
typedef std::unique_ptr<lm_gguf_context, lm_gguf_context_deleter> lm_gguf_context_ptr;

// ggml-alloc

struct lm_ggml_gallocr_deleter { void operator()(lm_ggml_gallocr_t galloc) { lm_ggml_gallocr_free(galloc); } };

typedef std::unique_ptr<lm_ggml_gallocr, lm_ggml_gallocr_deleter> lm_ggml_gallocr_ptr;

// ggml-backend

struct lm_ggml_backend_deleter        { void operator()(lm_ggml_backend_t backend)       { lm_ggml_backend_free(backend); } };
struct lm_ggml_backend_buffer_deleter { void operator()(lm_ggml_backend_buffer_t buffer) { lm_ggml_backend_buffer_free(buffer); } };
struct lm_ggml_backend_event_deleter  { void operator()(lm_ggml_backend_event_t event)   { lm_ggml_backend_event_free(event); } };
struct lm_ggml_backend_sched_deleter  { void operator()(lm_ggml_backend_sched_t sched)   { lm_ggml_backend_sched_free(sched); } };

typedef std::unique_ptr<lm_ggml_backend,        lm_ggml_backend_deleter>        lm_ggml_backend_ptr;
typedef std::unique_ptr<lm_ggml_backend_buffer, lm_ggml_backend_buffer_deleter> lm_ggml_backend_buffer_ptr;
typedef std::unique_ptr<lm_ggml_backend_event,  lm_ggml_backend_event_deleter>  lm_ggml_backend_event_ptr;
typedef std::unique_ptr<lm_ggml_backend_sched,  lm_ggml_backend_sched_deleter>  lm_ggml_backend_sched_ptr;

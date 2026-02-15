"""
Cactus FFI Python Bindings

Python bindings for Cactus Engine via FFI. Provides access to:
- Text completion with LLMs (including cloud handoff detection)
- Audio transcription with Whisper models
- Text, image, and audio embeddings
- RAG (Retrieval-Augmented Generation) queries
- Tool RAG (automatic tool selection based on query relevance)
- Streaming transcription
- Vector index for similarity search

Response Format:
All completion responses use a unified JSON format with all fields always present:
{
    "success": bool,        # True if generation succeeded
    "error": str|null,      # Error message if failed, null otherwise
    "cloud_handoff": bool,  # True if model recommends deferring to cloud
    "response": str|null,   # Generated text, null if cloud_handoff or error
    "function_calls": [],   # List of function calls if tools were used
    "confidence": float,    # Model confidence (1.0 - normalized_entropy)
    "time_to_first_token_ms": float,
    "total_time_ms": float,
    "prefill_tps": float,
    "decode_tps": float,
    "ram_usage_mb": float,
    "prefill_tokens": int,
    "decode_tokens": int,
    "total_tokens": int
}
"""
import ctypes
import json
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Cactus library not found at {_LIB_PATH}\n"
        f"Please build first: cactus build --python"
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
_lib.cactus_init.restype = ctypes.c_void_p

_lib.cactus_complete.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p
]
_lib.cactus_complete.restype = ctypes.c_int

_lib.cactus_transcribe.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_transcribe.restype = ctypes.c_int

_lib.cactus_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
]
_lib.cactus_embed.restype = ctypes.c_int

_lib.cactus_image_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_image_embed.restype = ctypes.c_int

_lib.cactus_audio_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_audio_embed.restype = ctypes.c_int

_lib.cactus_reset.argtypes = [ctypes.c_void_p]
_lib.cactus_reset.restype = None

_lib.cactus_stop.argtypes = [ctypes.c_void_p]
_lib.cactus_stop.restype = None

_lib.cactus_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_destroy.restype = None

_lib.cactus_get_last_error.argtypes = []
_lib.cactus_get_last_error.restype = ctypes.c_char_p

_lib.cactus_tokenize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.cactus_tokenize.restype = ctypes.c_int

_lib.cactus_score_window.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
_lib.cactus_score_window.restype = ctypes.c_int

_lib.cactus_rag_query.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_size_t
]
_lib.cactus_rag_query.restype = ctypes.c_int

_lib.cactus_stream_transcribe_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.cactus_stream_transcribe_start.restype = ctypes.c_void_p

_lib.cactus_stream_transcribe_process.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_process.restype = ctypes.c_int

_lib.cactus_stream_transcribe_stop.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_stop.restype = ctypes.c_int

_lib.cactus_index_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.cactus_index_init.restype = ctypes.c_void_p

_lib.cactus_index_add.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t
]
_lib.cactus_index_add.restype = ctypes.c_int

_lib.cactus_index_delete.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t
]
_lib.cactus_index_delete.restype = ctypes.c_int

_lib.cactus_index_query.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_index_query.restype = ctypes.c_int

_lib.cactus_index_compact.argtypes = [ctypes.c_void_p]
_lib.cactus_index_compact.restype = ctypes.c_int

_lib.cactus_index_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_index_destroy.restype = None


def cactus_init(model_path, corpus_dir=None, cache_index=False):
    """
    Initialize a model and return its handle.

    Args:
        model_path: Path to model weights directory
        corpus_dir: Optional path to RAG corpus directory for document Q&A
        cache_index: If True, load cached index if available; if False, always rebuild

    Returns:
        Model handle (opaque pointer) or None if initialization failed.
        Use cactus_get_last_error() to get error details.
    """
    return _lib.cactus_init(
        model_path.encode() if isinstance(model_path, str) else model_path,
        corpus_dir.encode() if corpus_dir else None,
        cache_index
    )


def cactus_complete(
    model,
    messages,
    tools=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_tokens=None,
    stop_sequences=None,
    include_stop_sequences=None,
    force_tools=False,
    tool_rag_top_k=None,
    confidence_threshold=None,
    callback=None
):
    """
    Run chat completion on a model.

    Args:
        model: Model handle from cactus_init
        messages: List of message dicts or JSON string
        tools: Optional list of tool definitions for function calling
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        max_tokens: Maximum tokens to generate
        stop_sequences: List of stop sequences
        include_stop_sequences: Include matched stop sequences in output (default: False)
        force_tools: Constrain output to tool call format
        tool_rag_top_k: Select top-k relevant tools via Tool RAG (default: 2, 0 = disabled)
        confidence_threshold: Minimum confidence for local generation (default: 0.7, triggers cloud_handoff when below)
        callback: Streaming callback fn(token, token_id, user_data)

    Returns:
        JSON string with unified response format (all fields always present):
        {
            "success": bool,        # True if generation succeeded
            "error": str|null,      # Error message if failed, null otherwise
            "cloud_handoff": bool,  # True if model confidence too low, should defer to cloud
            "response": str|null,   # Generated text, null if cloud_handoff or error
            "function_calls": [],   # List of function calls if tools were used
            "confidence": float,    # Model confidence (1.0 - normalized_entropy)
            "time_to_first_token_ms": float,
            "total_time_ms": float,
            "prefill_tps": float,
            "decode_tps": float,
            "ram_usage_mb": float,
            "prefill_tokens": int,
            "decode_tokens": int,
            "total_tokens": int
        }

        When cloud_handoff is True, the model confidence dropped below confidence_threshold
        and recommends deferring to a cloud-based model for better results.
    """
    if isinstance(messages, list):
        messages_json = json.dumps(messages)
    else:
        messages_json = messages

    tools_json = None
    if tools is not None:
        if isinstance(tools, list):
            tools_json = json.dumps(tools)
        else:
            tools_json = tools

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if stop_sequences is not None:
        options["stop_sequences"] = stop_sequences
    if include_stop_sequences is not None:
        options["include_stop_sequences"] = include_stop_sequences
    if force_tools:
        options["force_tools"] = True
    if tool_rag_top_k is not None:
        options["tool_rag_top_k"] = tool_rag_top_k
    if confidence_threshold is not None:
        options["confidence_threshold"] = confidence_threshold

    options_json = json.dumps(options) if options else None

    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_complete(
        model,
        messages_json.encode() if isinstance(messages_json, str) else messages_json,
        buf, len(buf),
        options_json.encode() if options_json else None,
        tools_json.encode() if tools_json else None,
        cb, None
    )
    return buf.value.decode("utf-8", errors="ignore")


def cactus_transcribe(model, audio_path, prompt="", callback=None):
    """
    Transcribe audio using a Whisper model.

    Args:
        model: Whisper model handle from cactus_init
        audio_path: Path to audio file (WAV format)
        prompt: Whisper prompt for language/task (e.g., "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
        callback: Optional streaming callback fn(token, token_id, user_data)

    Returns:
        JSON string with response: {"success": bool, "response": str, ...}
    """
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_transcribe(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        prompt.encode() if isinstance(prompt, str) else prompt,
        buf, len(buf),
        None, cb, None, None, 0
    )
    return buf.value.decode()


def cactus_embed(model, text, normalize=False):
    """
    Get text embeddings.

    Args:
        model: Model handle from cactus_init
        text: Text to embed
        normalize: L2-normalize embeddings (default: False)

    Returns:
        List of floats representing the embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_embed(
        model,
        text.encode() if isinstance(text, str) else text,
        buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize
    )
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """
    Get image embeddings from a VLM.

    Args:
        model: Model handle from cactus_init
        image_path: Path to image file

    Returns:
        List of floats representing the image embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_image_embed(
        model,
        image_path.encode() if isinstance(image_path, str) else image_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """
    Get audio embeddings from a Whisper model.

    Args:
        model: Whisper model handle from cactus_init
        audio_path: Path to audio file (WAV format)

    Returns:
        List of floats representing the audio embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_audio_embed(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_reset(model):
    """Reset model state (clear KV cache). Call between unrelated conversations."""
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Stop an ongoing generation (useful with streaming callbacks)."""
    _lib.cactus_stop(model)


def cactus_destroy(model):
    """Free model memory. Always call when done."""
    _lib.cactus_destroy(model)


def cactus_get_last_error():
    """Get the last error message, or None if no error."""
    result = _lib.cactus_get_last_error()
    return result.decode() if result else None


def cactus_tokenize(model, text: str):
    """
    Tokenize text.

    Args:
        model: Model handle from cactus_init
        text: Text to tokenize

    Returns:
        List of token IDs.
    """
    needed = ctypes.c_size_t(0)
    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        None,
        0,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize length query failed rc={rc}")

    n = needed.value
    arr = (ctypes.c_uint32 * n)()

    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        arr,
        n,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize fetch failed rc={rc}")

    return [arr[i] for i in range(n)]


def cactus_score_window(model, tokens, start, end, context):
    """
    Score a window of tokens for perplexity/log probability.

    Args:
        model: Model handle from cactus_init
        tokens: List of token IDs
        start: Start index of window to score
        end: End index of window to score
        context: Context size for scoring

    Returns:
        Dict with "success", "logprob", and "tokens" keys.
    """
    buf = ctypes.create_string_buffer(4096)
    n = len(tokens)
    arr = (ctypes.c_uint32 * n)(*tokens)

    _lib.cactus_score_window(
        model,
        arr,
        n,
        start,
        end,
        context,
        buf,
        len(buf),
    )
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_rag_query(model, query, top_k=5):
    """
    Query RAG corpus for relevant text chunks.

    Args:
        model: Model handle (must have been initialized with corpus_dir)
        query: Query text
        top_k: Number of chunks to retrieve (default: 5)

    Returns:
        List of dicts with "score" and "text" keys, or empty list on error.
    """
    buf = ctypes.create_string_buffer(65536)
    result = _lib.cactus_rag_query(
        model,
        query.encode() if isinstance(query, str) else query,
        buf, len(buf), top_k
    )
    if result != 0:
        return []
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_stream_transcribe_start(model, options=None):
    """
    Initialize streaming transcription session.

    Args:
        model: Whisper model handle from cactus_init
        options: Optional JSON string with options

    Returns:
        Stream handle for use with other stream_transcribe functions.
    """
    return _lib.cactus_stream_transcribe_start(
        model,
        options.encode() if options else None
    )


def cactus_stream_transcribe_process(stream, pcm_data):
    """
    Process audio data and return transcription.

    Args:
        stream: Stream handle from cactus_stream_transcribe_start
        pcm_data: PCM audio data as bytes or list of uint8

    Returns:
        JSON string with transcription result.
    """
    if isinstance(pcm_data, bytes):
        arr = (ctypes.c_uint8 * len(pcm_data)).from_buffer_copy(pcm_data)
    else:
        arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)

    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_process(stream, arr, len(arr), buf, len(buf))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_stop(stream):
    """
    Finalize streaming transcription and get final result.

    Args:
        stream: Stream handle from cactus_stream_transcribe_start

    Returns:
        JSON string with final transcription result.
    """
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_stop(stream, buf, len(buf))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_index_init(index_dir, embedding_dim):
    """
    Initialize a vector index.

    Args:
        index_dir: Path to directory for index storage
        embedding_dim: Dimension of embedding vectors

    Returns:
        Index handle (opaque pointer) or None if initialization failed.
    """
    return _lib.cactus_index_init(
        index_dir.encode() if isinstance(index_dir, str) else index_dir,
        embedding_dim
    )


def cactus_index_add(index, ids, documents, embeddings, metadatas=None):
    """
    Add documents to the index.

    Args:
        index: Index handle from cactus_index_init
        ids: List of integer document IDs
        documents: List of document strings
        embeddings: List of embedding vectors (list of floats each)
        metadatas: Optional list of metadata strings

    Returns:
        0 on success, -1 on error.
    """
    count = len(ids)
    embedding_dim = len(embeddings[0]) if embeddings else 0

    ids_arr = (ctypes.c_int * count)(*ids)

    docs_arr = (ctypes.c_char_p * count)()
    for i, doc in enumerate(documents):
        docs_arr[i] = doc.encode() if isinstance(doc, str) else doc

    meta_arr = None
    if metadatas:
        meta_arr = (ctypes.c_char_p * count)()
        for i, meta in enumerate(metadatas):
            meta_arr[i] = meta.encode() if isinstance(meta, str) else meta

    emb_ptrs = (ctypes.POINTER(ctypes.c_float) * count)()
    emb_arrays = []
    for i, emb in enumerate(embeddings):
        arr = (ctypes.c_float * len(emb))(*emb)
        emb_arrays.append(arr)
        emb_ptrs[i] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

    return _lib.cactus_index_add(
        index,
        ids_arr,
        docs_arr,
        meta_arr,
        emb_ptrs,
        count,
        embedding_dim
    )


def cactus_index_delete(index, ids):
    """
    Delete documents from the index.

    Args:
        index: Index handle from cactus_index_init
        ids: List of document IDs to delete

    Returns:
        0 on success, -1 on error.
    """
    count = len(ids)
    ids_arr = (ctypes.c_int * count)(*ids)
    return _lib.cactus_index_delete(index, ids_arr, count)


def cactus_index_query(index, embedding, top_k=5, options=None):
    """
    Query the index for similar documents.

    Args:
        index: Index handle from cactus_index_init
        embedding: Query embedding vector (list of floats)
        top_k: Number of results to return (default: 5)
        options: Optional JSON string with query options

    Returns:
        List of dicts with "id" and "score" keys.
    """
    embedding_dim = len(embedding)

    emb_arr = (ctypes.c_float * embedding_dim)(*embedding)
    emb_ptr = ctypes.cast(emb_arr, ctypes.POINTER(ctypes.c_float))
    emb_ptr_ptr = ctypes.pointer(emb_ptr)

    id_buffer = (ctypes.c_int * top_k)()
    score_buffer = (ctypes.c_float * top_k)()

    id_ptr = ctypes.cast(id_buffer, ctypes.POINTER(ctypes.c_int))
    score_ptr = ctypes.cast(score_buffer, ctypes.POINTER(ctypes.c_float))

    id_size = ctypes.c_size_t(top_k)
    score_size = ctypes.c_size_t(top_k)

    id_ptr_ptr = ctypes.pointer(id_ptr)
    score_ptr_ptr = ctypes.pointer(score_ptr)

    options_encoded = options.encode() if options else None

    result = _lib.cactus_index_query(
        index,
        emb_ptr_ptr,
        1,
        embedding_dim,
        options_encoded,
        id_ptr_ptr,
        ctypes.byref(id_size),
        score_ptr_ptr,
        ctypes.byref(score_size)
    )

    if result < 0:
        return []

    return [
        {"id": id_buffer[i], "score": score_buffer[i]}
        for i in range(id_size.value)
    ]


def cactus_index_compact(index):
    """
    Compact the index to optimize storage and query performance.

    Args:
        index: Index handle from cactus_index_init

    Returns:
        0 on success, -1 on error.
    """
    return _lib.cactus_index_compact(index)


def cactus_index_destroy(index):
    """Free index resources. Always call when done."""
    _lib.cactus_index_destroy(index)

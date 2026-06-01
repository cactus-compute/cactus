# DeepSeek-V4 Encoding

This document describes the prompt encoding format used by DeepSeek-V4 series models. The encoding handles multi-turn conversations, tool calling, extended thinking (reasoning), and quick instruction tasks.

A self-contained reference implementation is provided in `encoding_dsv4.py`.

## Quick Start

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

# Encode a conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
prompt = encode_messages(messages, thinking_mode="thinking")
# => "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"

# Parse model output back to structured message
completion = "Simple arithmetic.</think>2 + 2 = 4.<｜end▁of▁sentence｜>"
parsed = parse_message_from_completion_text(completion, thinking_mode="thinking")
# => {"role": "assistant", "reasoning_content": "Simple arithmetic.", "content": "2 + 2 = 4.", "tool_calls": []}
```

> **Note:** The `parse_message_from_completion_text` function is designed to handle well-formatted model output only. It does not attempt to correct or recover from malformed output that the model might occasionally generate. For production use, additional error handling is recommended.

## Message Format

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<｜begin▁of▁sentence｜>` | Beginning of sequence (BOS) |
| `<｜end▁of▁sentence｜>` | End of assistant turn (EOS) |
| `<｜User｜>` | User turn prefix |
| `<｜Assistant｜>` | Assistant turn prefix |
| `<｜latest_reminder｜>` | Latest reminder (date, locale, etc.) |
| `<think>` / `</think>` | Reasoning block delimiters |
| `｜DSML｜` | DSML markup token |

### Roles

The encoding supports the following message roles: `system`, `user`, `assistant`, `tool`, `latest_reminder`, and `developer`.

> **Note on the `developer` role:** The `developer` role is used exclusively in the internal search agent pipeline. It is not needed for general-purpose chat or tool-calling tasks, and the official API does not accept messages with this role.

### Basic Chat

A simple multi-turn conversation is encoded as:

```
<｜begin▁of▁sentence｜>{system_prompt}
<｜User｜>{user_message}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
<｜User｜>{user_message_2}<｜Assistant｜></think>{response_2}<｜end▁of▁sentence｜>
```

- The BOS token is prepended at the very beginning of the conversation.
- In **chat mode** (`thinking_mode="chat"`), `</think>` is placed right after `<｜Assistant｜>` to immediately close the thinking block, so the model generates content directly.

### Interleaved Thinking Mode

In **thinking mode** (`thinking_mode="thinking"`), the model produces explicit reasoning inside `<think>...</think>` blocks before responding.

```
<｜begin▁of▁sentence｜>{system_prompt}
<｜User｜>{message}<｜Assistant｜><think>{reasoning}</think>{response}<｜end▁of▁sentence｜>
```

The `drop_thinking` parameter (default `True`) controls whether reasoning from earlier turns is preserved:

- **Without tools**: `drop_thinking` takes effect. Reasoning content from assistant turns **before** the last user message is stripped. Only the final assistant turn retains its `<think>...</think>` block.
- **With tools** (on system or developer message): `drop_thinking` is automatically disabled. All turns retain their reasoning, because tool-calling conversations require full context for the model to track multi-step reasoning across tool calls.

### Tool Calling (DSML Format)

Tools are defined on the `system` or `developer` message via the `tools` field (OpenAI-compatible format). When tools are present, the following schema block is injected into the system/user prompt:

```
## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_definitions_json}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
```

An actual tool call in the assistant turn looks like:

```xml
<｜DSML｜tool_calls>
<｜DSML｜invoke name="function_name">
<｜DSML｜parameter name="param" string="true">string_value</｜DSML｜parameter>
<｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>
```

- `string="true"`: the parameter value is a raw string.
- `string="false"`: the parameter value is JSON (number, boolean, array, object).

Tool execution results are wrapped in `<tool_result>` tags within user messages:

```
<｜User｜><tool_result>{result_json}</tool_result><｜Assistant｜><think>...
```

When multiple tool results are present, they are sorted by the order of the corresponding `tool_calls` in the preceding assistant message.

### Reasoning Effort

When `reasoning_effort="max"` is set, a special prefix is prepended at the very beginning of the prompt (before the system message) to instruct the model to maximize its reasoning depth:

```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.
```

### Quick Instruction Special Tokens

Quick instruction tokens are used for auxiliary classification and generation tasks. They are appended to messages via the `"task"` field to trigger specialized model behavior for a single-token or short-form output.

| Special Token | Description | Format |
|:---|:---|:---|
| `<｜action｜>` | Determines whether the user prompt requires a web search or can be answered directly. | `...<｜User｜>{prompt}<｜Assistant｜><think><｜action｜>` |
| `<｜title｜>` | Generates a concise conversation title after the first assistant response. | `...<｜Assistant｜>{response}<｜end▁of▁sentence｜><｜title｜>` |
| `<｜query｜>` | Generates search queries for the user prompt. | `...<｜User｜>{prompt}<｜query｜>` |
| `<｜authority｜>` | Classifies the user prompt's demand for source authoritativeness. | `...<｜User｜>{prompt}<｜authority｜>` |
| `<｜domain｜>` | Identifies the domain of the user prompt. | `...<｜User｜>{prompt}<｜domain｜>` |
| `<｜extracted_url｜>` `<｜read_url｜>` | Determines whether each URL in the user prompt should be fetched and read. | `...<｜User｜>{prompt}<｜extracted_url｜>{url}<｜read_url｜>` |

Usage in message format:

- **`action`** on a user message: the `<｜action｜>` token is placed after the assistant prefix and thinking token, triggering a routing decision (e.g., "Search" or "Answer").
- **Other tasks** (`query`, `authority`, `domain`, `read_url`) on a user message: the task token is appended directly after the user content.
- **`title`** on an assistant message: the `<｜title｜>` token is appended after the assistant's EOS. The next assistant message provides the generated title.

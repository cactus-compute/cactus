class ChatMessage {
  final String role;
  final String content;
  final double? tokensPerSecond;

  ChatMessage({required this.role, required this.content, this.tokensPerSecond});

  Map<String, String> toJson() => {
        'role': role,
        'content': content,
      };
}

const String defaultChatMLTemplate = """
{% for message in messages %}
  {% if message.role == 'system' %}
    {{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
  {% elif message.role == 'user' %}
    {{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
  {% elif message.role == 'assistant' %}
    {{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant\n' }}
{% endif %}
""";
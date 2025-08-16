import type { CactusOAICompatibleMessage } from "./chat";

let _cactusToken: string | null = null;

export function setCactusToken(token: string | null): void {
  _cactusToken = token;
}

export async function getVertexAIEmbedding(text: string): Promise<number[]> {
  text = text
  throw new Error('Remote embedding is not currently supported. The Cactus library is in active development - if you need this functionality, please contact us at founders@cactuscompute.com');
}

export async function getVertexAICompletion(
  messages: CactusOAICompatibleMessage[],
  imageData?: string,
  imagePath?: string,
  mimeType?: string,
): Promise<string> {
  if (_cactusToken === null) {
    throw new Error('CactusToken not set. Please call CactusVLM.init with cactusToken parameter.');
  }
  const endpoint = 'https://openrouter.ai/api/v1/chat/completions';

  const headers = {
    'Authorization': `Bearer ${_cactusToken}`,
    'Content-Type': 'application/json',
  };

  const requestBody = {
    model: 'google/gemini-2.5-flash-lite',
    messages: messages,
  };

  let imageUrl = ''
  if (imageData) {
    imageUrl = `data:${mimeType || 'image/jpeg'};base64,${imageData}`
  } else if (imagePath) {
    const RNFS = require('react-native-fs');
    const base64Data = await RNFS.readFile(imagePath, 'base64');
    imageUrl = `data:${mimeType || detectMimeType(imagePath)};base64,${base64Data}`
  }

  if (imageUrl) {
    if (requestBody.messages[requestBody.messages.length - 1]?.role === 'user') {
      requestBody.messages[requestBody.messages.length - 1] = {
        role: 'user',
        content: [
          {
            type: 'text',
            text: requestBody.messages[requestBody.messages.length - 1]?.content || ''
          },
          {
            type: 'image_url',
            image_url: {
              url: imageUrl
            }
          }
        ]
      }
    }else{
      console.warn('Image data provided but message is not a user message: ', requestBody.messages);
    }
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody),
  });

  if (response.status === 401) {
    _cactusToken = null;
    throw new Error('Authentication failed. Please update your cactusToken.');
  } else if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const responseBody = await response.json();
  const responseText = responseBody.choices[0].message.content;

  return responseText;
}

export async function getTextCompletion(messages: CactusOAICompatibleMessage[]): Promise<string> {
  return getVertexAICompletion(messages);
}

export async function getVisionCompletion(messages: CactusOAICompatibleMessage[], imagePath: string): Promise<string> {
  return getVertexAICompletion(messages, undefined, imagePath);
}

export async function getVisionCompletionFromData(messages: CactusOAICompatibleMessage[], imageData: string, mimeType?: string): Promise<string> {
  return getVertexAICompletion(messages, imageData, undefined, mimeType);
}

function detectMimeType(filePath: string): string {
  const extension = filePath.toLowerCase().split('.').pop();
  switch (extension) {
    case 'jpg':
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'gif':
      return 'image/gif';
    case 'webp':
      return 'image/webp';
    default:
      return 'image/jpeg';
  }
} 
import { openAICompletionsApi } from "../api/openai-completions.lazy.ts";
import type { ApiKeyAuth } from "../auth/types.ts";
import { createProvider, type Provider } from "../models.ts";
import type { Model } from "../types.ts";

const DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1";
const DEFAULT_CONTEXT_WINDOW = 32768;
const DEFAULT_MAX_TOKENS = 8192;
const MAX_CONTEXT_WINDOW = 8_000_000;

function sanitizeContextWindow(cw: unknown): number {
	return typeof cw === "number" && Number.isFinite(cw) && cw > 0 && cw <= MAX_CONTEXT_WINDOW
		? Math.floor(cw)
		: DEFAULT_CONTEXT_WINDOW;
}

export function cactusBaseUrl(): string {
	return process.env.CACTUS_BASE_URL?.trim().replace(/\/+$/, "") || DEFAULT_BASE_URL;
}

function cactusAuth(): ApiKeyAuth {
	return {
		name: "Cactus (local, no auth)",
		resolve: async () => ({ auth: { apiKey: "cactus" }, source: "local server" }),
	};
}

function isVisionModel(id: string, modelType: string): boolean {
	const s = `${id} ${modelType}`.toLowerCase();
	return /(?:^|[^a-z])vl(?:[^a-z]|$)|gemma-?4|gemma-?3n|qwen3\.5/.test(s);
}

function isThinkingModel(id: string, modelType: string): boolean {
	const s = `${id} ${modelType}`.toLowerCase();
	return /gemma-?4|qwen3/.test(s);
}

interface OpenAIModelObject {
	id?: string;
	context_window?: number;
	model_type?: string;
}
interface OpenAIModelsResponse {
	data?: OpenAIModelObject[];
}

export function cactusModelFromId(
	baseUrl: string,
	id: string,
	contextWindow: number = DEFAULT_CONTEXT_WINDOW,
	modelType = "",
): Model<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "cactus",
		baseUrl,
		reasoning: isThinkingModel(id, modelType),
		input: isVisionModel(id, modelType) ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: sanitizeContextWindow(contextWindow),
		maxTokens: DEFAULT_MAX_TOKENS,
	};
}

export async function fetchCactusModels(baseUrl: string = cactusBaseUrl()): Promise<Model<"openai-completions">[]> {
	const res = await fetch(`${baseUrl}/models`);
	if (!res.ok) throw new Error(`Cactus model discovery failed: HTTP ${res.status}`);
	const body = (await res.json()) as OpenAIModelsResponse;
	const data = Array.isArray(body?.data) ? body.data : [];
	return data
		.filter((m): m is OpenAIModelObject & { id: string } => typeof m?.id === "string" && m.id.length > 0)
		.map((m) =>
			cactusModelFromId(
				baseUrl,
				m.id,
				typeof m.context_window === "number" ? m.context_window : DEFAULT_CONTEXT_WINDOW,
				typeof m.model_type === "string" ? m.model_type : "",
			),
		);
}

export function cactusProvider(): Provider<"openai-completions"> {
	const baseUrl = cactusBaseUrl();
	return createProvider({
		id: "cactus",
		name: "Cactus",
		baseUrl,
		auth: { apiKey: cactusAuth() },
		models: [],
		refreshModels: () => fetchCactusModels(baseUrl),
		api: openAICompletionsApi(),
	});
}

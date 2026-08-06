export type { Static, TSchema } from "typebox";
export { Type } from "typebox";

// Core only, side-effect free: no generated catalogs, no provider factories,
// no api-registry, no OAuth implementations, no compat. Provider factories
// live under "@earendil-works/pi-ai/providers/*", API implementations under
// "@earendil-works/pi-ai/api/*", the old global API under
// "@earendil-works/pi-ai/compat".
export * from "./api/lazy.ts";
export type { OpenAICompletionsOptions } from "./api/openai-completions.ts";
export * from "./auth/context.ts";
export * from "./auth/credential-store.ts";
export * from "./auth/helpers.ts";
export * from "./auth/types.ts";
export * from "./models.ts";
export * from "./providers/faux.ts";
export * from "./session-resources.ts";
export * from "./types.ts";
export * from "./utils/diagnostics.ts";
export * from "./utils/event-stream.ts";
export * from "./utils/json-parse.ts";
export type {
	OAuthAuthInfo,
	OAuthCredentials,
	OAuthDeviceCodeInfo,
	OAuthLoginCallbacks,
	OAuthPrompt,
	OAuthProvider,
	OAuthProviderId,
	OAuthProviderInfo,
	OAuthProviderInterface,
	OAuthSelectOption,
	OAuthSelectPrompt,
} from "./utils/oauth/types.ts";
export * from "./utils/overflow.ts";
export * from "./utils/typebox-helpers.ts";
export * from "./utils/validation.ts";

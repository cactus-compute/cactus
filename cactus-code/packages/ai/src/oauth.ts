import type {
	OAuthCredentials,
	OAuthProviderId,
	OAuthProviderInfo,
	OAuthProviderInterface,
} from "./utils/oauth/types.ts";

export * from "./utils/oauth/types.ts";

const oauthProviderRegistry = new Map<string, OAuthProviderInterface>();

export function getOAuthProvider(id: OAuthProviderId): OAuthProviderInterface | undefined {
	return oauthProviderRegistry.get(id);
}

export function registerOAuthProvider(provider: OAuthProviderInterface): void {
	oauthProviderRegistry.set(provider.id, provider);
}

export function unregisterOAuthProvider(id: string): void {
	oauthProviderRegistry.delete(id);
}

export function resetOAuthProviders(): void {
	oauthProviderRegistry.clear();
}

export function getOAuthProviders(): OAuthProviderInterface[] {
	return Array.from(oauthProviderRegistry.values());
}

export function getOAuthProviderInfoList(): OAuthProviderInfo[] {
	return getOAuthProviders().map((p) => ({ id: p.id, name: p.name, available: true }));
}

export async function refreshOAuthToken(
	providerId: OAuthProviderId,
	credentials: OAuthCredentials,
): Promise<OAuthCredentials> {
	const provider = getOAuthProvider(providerId);
	if (!provider) throw new Error(`Unknown OAuth provider: ${providerId}`);
	return provider.refreshToken(credentials);
}

export async function getOAuthApiKey(
	providerId: OAuthProviderId,
	credentials: Record<string, OAuthCredentials>,
): Promise<{ newCredentials: OAuthCredentials; apiKey: string } | null> {
	const provider = getOAuthProvider(providerId);
	if (!provider) throw new Error(`Unknown OAuth provider: ${providerId}`);
	let creds = credentials[providerId];
	if (!creds) return null;
	if (Date.now() >= creds.expires) {
		try {
			creds = await provider.refreshToken(creds);
		} catch (_error) {
			throw new Error(`Failed to refresh OAuth token for ${providerId}`);
		}
	}
	return { newCredentials: creds, apiKey: provider.getApiKey(creds) };
}

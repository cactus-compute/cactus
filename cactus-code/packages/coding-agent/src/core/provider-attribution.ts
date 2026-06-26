import type { Api, Model, ProviderHeaders } from "@earendil-works/pi-ai";
import type { SettingsManager } from "./settings-manager.ts";

export function mergeProviderAttributionHeaders(
	_model: Model<Api>,
	_settingsManager: SettingsManager,
	_sessionId: string | undefined,
	...headerSources: Array<ProviderHeaders | undefined>
): ProviderHeaders | undefined {
	const merged: ProviderHeaders = {};
	for (const headers of headerSources) {
		if (headers) {
			Object.assign(merged, headers);
		}
	}
	return Object.keys(merged).length > 0 ? merged : undefined;
}

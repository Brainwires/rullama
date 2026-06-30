import { usePersistedState } from "@/lib/persisted";

/**
 * Persisted Cloud-tab settings (the non-secret bits). The API keys themselves
 * are NOT kept here — they live encrypted in the OPFS/WebCrypto vault
 * (`lib/cloud/keyvault`), read on demand at request time. This hook only holds:
 *
 *   - `cloudBaseOverride` — optional proxy base URL for the power-user "point at
 *      my own Cloudflare Worker" path. Empty → same-origin `/api/cloud/*`.
 *   - `cloudConsented`    — the one-time acknowledgement that cloud chat sends
 *      messages + the API key off-device (gates the first cloud load/send).
 */
export function useCloudSettings() {
    const [cloudBaseOverride, setCloudBaseOverride] = usePersistedState<string>("rullama:cloudBaseOverride", "");
    const [cloudConsented, setCloudConsented] = usePersistedState<boolean>("rullama:cloudConsented", false);

    return {
        cloudBaseOverride, setCloudBaseOverride,
        cloudConsented, setCloudConsented,
    };
}

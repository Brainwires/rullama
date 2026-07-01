// Boot-time update check.
//
// The bundle's `__APP_VERSION__` constant is injected by Vite from
// `public/version.json` at build time (see `vite.config.ts`). At
// runtime we re-fetch `/version.json` from the server with a short
// timeout; a mismatch means a newer build has been deployed and the
// app should surface an "Update available" banner.
//
// Offline guard: if `navigator.onLine` reports false OR the fetch
// fails/times out, return null and let the caller treat it as
// "no update detected" — never blocks boot, never errors loudly.

declare const __APP_VERSION__: string;

/** Version string baked into the running JS bundle. Compare against
 *  the server-fetched `version` field; equal → already on the latest. */
export const BUNDLED_VERSION: string = __APP_VERSION__;

export interface ServerVersion {
    version: string;
    builtAt: string;
    commit:  string;
}

const DEFAULT_TIMEOUT_MS = 3_000;

export async function fetchServerVersion(timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<ServerVersion | null> {
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
        return null;
    }
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const r = await fetch("/version.json", {
            signal: ctrl.signal,
            // Belt + suspenders over the nginx `no-store` header so a
            // misbehaving intermediate cache can't serve us a stale
            // version.json.
            cache: "no-store",
        });
        if (!r.ok) return null;
        const j = await r.json() as Partial<ServerVersion>;
        if (typeof j?.version !== "string") return null;
        return {
            version: j.version,
            builtAt: typeof j.builtAt === "string" ? j.builtAt : "",
            commit:  typeof j.commit  === "string" ? j.commit  : "",
        };
    } catch {
        return null;
    } finally {
        clearTimeout(t);
    }
}

/** True when the server's version differs from what's baked into the
 *  running bundle. `null` server (offline/timeout/parse fail) → false:
 *  we never claim an update without positive evidence. */
export function isUpdateAvailable(server: ServerVersion | null, bundled: string = BUNDLED_VERSION): boolean {
    if (!server) return false;
    if (!server.version) return false;
    if (server.version === bundled) return false;
    if (bundled === "dev") return false; // dev build — never prompt to "update" away from itself
    return true;
}

/** Friendlier label for the banner: trim the commit tail off the version
 *  string so users see something like `v20260520-143012` instead of the
 *  full `20260520-143012-abc1234`. */
export function shortVersionLabel(v: string): string {
    if (!v) return "";
    // Strip a trailing -xxxxxxxx commit segment if present.
    return v.replace(/-[a-fA-F0-9]{6,12}$/, "");
}

// localStorage key for "user clicked Later on this version" — keyed
// by the SERVER version we deferred, so we don't keep prompting on
// every reload. Cleared automatically when the bundle finally catches
// up (`BUNDLED_VERSION === dismissed`).
const DISMISS_KEY = "rullama:dismissedUpdateVersion";

// localStorage can throw in Safari private browsing and on quota
// exhaustion. We don't propagate those errors — the dismiss flag is
// a UX nicety, not load-bearing — but log so it's diagnosable.
export function getDismissedVersion(): string | null {
    try { return localStorage.getItem(DISMISS_KEY); }
    catch (e) { console.warn("[rullama] localStorage read failed (getDismissedVersion):", e); return null; }
}

export function setDismissedVersion(v: string): void {
    try { localStorage.setItem(DISMISS_KEY, v); }
    catch (e) { console.warn("[rullama] localStorage write failed (setDismissedVersion):", e); }
}

export function clearDismissedVersion(): void {
    try { localStorage.removeItem(DISMISS_KEY); }
    catch (e) { console.warn("[rullama] localStorage remove failed (clearDismissedVersion):", e); }
}

/** True if the user already clicked "Later" on exactly this server
 *  version (and we haven't yet shipped a build that matches it). */
export function isDismissed(server: ServerVersion | null): boolean {
    if (!server) return false;
    const d = getDismissedVersion();
    if (!d) return false;
    // If we've actually updated to the dismissed version, clear it.
    if (d === BUNDLED_VERSION) {
        clearDismissedVersion();
        return false;
    }
    return d === server.version;
}

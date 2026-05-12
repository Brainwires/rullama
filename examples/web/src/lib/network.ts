// Best-effort network type detection via the Network Information API.
//
// Browser support reality (2026):
//   - `saveData`        — exposed broadly (Chromium + WebKit). On iOS it
//                         reflects "Low Data Mode"; on Android, "Data Saver".
//                         The strongest signal we have: the user has
//                         explicitly opted into byte-conscious behavior.
//   - `type`            — only Android Chrome reports a real value (e.g.
//                         "cellular", "wifi"). iOS Safari + every desktop
//                         leave it `undefined`. Treat presence as a hard yes.
//   - `effectiveType`   — `"slow-2g" | "2g" | "3g" | "4g"`, throughput-based
//                         not transport-based. Slow WiFi reads as "3g";
//                         strong LTE reads as "4g". Useful as a tiebreaker.
//
// There is no way to *prove* the device is on cellular from JS on iOS.
// We surface what we can and let the user decide.

interface NetworkInformationLike {
    saveData?:      boolean;
    type?:          string;
    effectiveType?: string;
}

function getConnection(): NetworkInformationLike | undefined {
    if (typeof navigator === "undefined") return undefined;
    type WithConn = Navigator & { connection?: NetworkInformationLike };
    return (navigator as WithConn).connection;
}

export interface NetworkHint {
    saveData:        boolean;
    type:            string | null;     // "cellular" | "wifi" | …
    effectiveType:   string | null;     // "slow-2g" | "2g" | "3g" | "4g"
    /** True when we have any positive signal that the link might be metered
     *  or slow: Low Data Mode on, transport explicitly cellular, or a 2G/3G
     *  effective tier. False is "we don't know" — not "definitely WiFi." */
    metered:         boolean;
    /** Human-readable summary, populated only when `metered` is true. */
    reason:          string | null;
}

export function getNetworkHint(): NetworkHint {
    const c = getConnection();
    const saveData      = !!c?.saveData;
    const type          = c?.type ?? null;
    const effectiveType = c?.effectiveType ?? null;

    const reasons: string[] = [];
    if (saveData) reasons.push("Low Data Mode is on");
    if (type === "cellular") reasons.push("you're on cellular");
    if (effectiveType && effectiveType !== "4g") reasons.push(`slow connection (${effectiveType})`);

    return {
        saveData,
        type,
        effectiveType,
        metered: reasons.length > 0,
        reason:  reasons.length > 0 ? reasons.join(" + ") : null,
    };
}

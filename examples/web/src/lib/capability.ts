// Device-capability probe + tiering.
//
// Why this exists: old devices (e.g. iPhone 7 — no WebGPU at all) boot-loop
// the heavy app. We classify the device into a tier *before* booting the
// engine and hard-block below the iPhone-16e-class minimum with a clean
// screen, so the boot-loop can't happen.
//
// **Load-bearing API constraint:** WebGPU exposes NO total-VRAM figure. The
// only readable signals are `navigator.gpu` presence, `adapter.limits`
// (a capability proxy — a 12 GB and a 24 GB card often report the SAME
// limits), `navigator.deviceMemory` (system RAM, rounded + capped at 8,
// Chromium-only), and sometimes `adapter.info` (hidden on Safari). So the
// **minimum gate is reliably detectable**, but **Premium (24 GB) is NOT** —
// it's a conservative auto-guess (errs low) OR a manual Settings override.

import { useEffect, useState } from "react";
import { usePersistedState } from "@/lib/persisted";

/** Persisted manual Premium override (Settings → "High-VRAM GPU (24GB+)"). */
export const HIGH_VRAM_OVERRIDE_KEY = "highVramOverride"; // usePersistedState prefixes "rullama:"

export type DeviceTier = "unsupported" | "mobile" | "desktop" | "premium";

/** Raw probe results — also handy for diagnostics (Settings/EnvironmentStatus). */
export interface GpuProbe {
    hasGpu:           boolean;   // navigator.gpu present
    adapterOk:        boolean;   // requestAdapter() returned an adapter
    maxBufferSize:    number;    // adapter.limits.maxBufferSize (bytes)
    maxStorageBinding: number;   // adapter.limits.maxStorageBufferBindingSize (bytes)
    deviceMemoryGB:   number;    // navigator.deviceMemory (rounded, capped 8); 8 when unknown
    isMobileUA:       boolean;
    vendor?:          string;    // adapter.info.vendor (often empty on Safari)
    architecture?:    string;    // adapter.info.architecture
    description?:     string;    // adapter.info.description
    error?:           string;    // requestAdapter throw, if any
}

// iPhone-16e-class single-buffer floor. A device that can't hold a 512 MB
// weight tile can't run the inference forward path; below this we hard-block
// (this is also what catches WebGPU-less devices, which report hasGpu=false).
const MIN_MAX_BUFFER = 512 * 1024 * 1024;
// Conservative auto-Premium guess (errs LOW on purpose — real 24 GB cards
// that don't advertise large limits just won't auto-detect; the user flips
// the manual toggle). Requires a desktop UA, a large advertised buffer, and
// the deviceMemory cap maxed out.
const PREMIUM_AUTO_MAX_BUFFER = 4 * 1024 * 1024 * 1024;
const PREMIUM_AUTO_MEM_GB = 8;

const isMobileUA = (): boolean =>
    typeof navigator !== "undefined" && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

/** Probe the GPU once. Never throws — failures land in the returned struct. */
export async function probeGpu(): Promise<GpuProbe> {
    const deviceMemoryGB = (typeof navigator !== "undefined"
        ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory
        : undefined) ?? 8;
    const base: GpuProbe = {
        hasGpu: false,
        adapterOk: false,
        maxBufferSize: 0,
        maxStorageBinding: 0,
        deviceMemoryGB,
        isMobileUA: isMobileUA(),
    };

    const gpu = typeof navigator !== "undefined"
        ? (navigator as Navigator & { gpu?: GPU }).gpu
        : undefined;
    if (!gpu) return base;
    base.hasGpu = true;

    let adapter: GPUAdapter | null = null;
    try {
        adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    } catch (err) {
        base.error = String(err);
        return base;
    }
    if (!adapter) return base;

    base.adapterOk = true;
    base.maxBufferSize = adapter.limits.maxBufferSize ?? 0;
    base.maxStorageBinding = adapter.limits.maxStorageBufferBindingSize ?? 0;
    // adapter.info is a getter in newer specs; requestAdapterInfo() in older.
    // Both are best-effort and usually empty on Safari.
    try {
        const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
        if (info) {
            base.vendor = info.vendor;
            base.architecture = info.architecture;
            base.description = info.description;
        }
    } catch { /* info unavailable */ }
    return base;
}

/** Map a probe (+ manual override) to a tier. */
export function tierFromProbe(p: GpuProbe, highVramOverride: boolean): DeviceTier {
    if (!p.hasGpu || !p.adapterOk || p.maxBufferSize < MIN_MAX_BUFFER) return "unsupported";
    if (p.isMobileUA) return "mobile";
    // Desktop. Premium = manual override OR a conservative auto-guess.
    const autoPremium =
        p.maxBufferSize >= PREMIUM_AUTO_MAX_BUFFER && p.deviceMemoryGB >= PREMIUM_AUTO_MEM_GB;
    if (highVramOverride || autoPremium) return "premium";
    return "desktop";
}

/** One-shot: probe + tier. */
export async function detectTier(highVramOverride: boolean): Promise<{ tier: DeviceTier; probe: GpuProbe }> {
    const probe = await probeGpu();
    return { tier: tierFromProbe(probe, highVramOverride), probe };
}

export interface DeviceTierState {
    tier:    DeviceTier | null; // null while the async probe is in flight
    probe:   GpuProbe | null;
    loading: boolean;
}

/**
 * Hook: resolves the device tier, re-resolving when the manual High-VRAM
 * override flips. The override itself is read via usePersistedState so a
 * Settings toggle re-renders consumers. `tier` is null until the first
 * probe lands — callers should treat null as "still checking" (don't boot
 * the engine yet, don't render the Unsupported screen yet).
 */
export function useDeviceTier(): DeviceTierState {
    const [override] = usePersistedState<boolean>(HIGH_VRAM_OVERRIDE_KEY, false);
    const [state, setState] = useState<DeviceTierState>({ tier: null, probe: null, loading: true });

    useEffect(() => {
        let cancelled = false;
        setState((s) => ({ ...s, loading: true }));
        void detectTier(override).then(({ tier, probe }) => {
            if (!cancelled) setState({ tier, probe, loading: false });
        });
        return () => { cancelled = true; };
    }, [override]);

    return state;
}

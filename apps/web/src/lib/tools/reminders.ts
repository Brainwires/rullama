// set_timer / set_reminder — real, via setTimeout + the Notification API. Runs
// on the main thread (where executeTool is called), so notifications work.
// Scope: RELATIVE durations only ("in 10 minutes") — absolute clock times need
// a persistent scheduler we don't have. Fires only while the tab is open; the
// result says so.

import type { ToolDef, ToolRunResult } from "@/lib/tools/types";

/** Parse "30 seconds" / "5 min" / "2 hours" / "1h30m" / "90" → seconds, or null. */
function parseDurationSec(s: string): number | null {
    const t = s.trim().toLowerCase();
    if (!t) return null;
    let total = 0;
    let matched = false;
    const re = /(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)?/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(t)) !== null) {
        const n = parseFloat(m[1]);
        if (Number.isNaN(n)) continue;
        matched = true;
        const unit = m[2] ?? "";
        if (unit.startsWith("h")) total += n * 3600;
        else if (unit.startsWith("m")) total += n * 60;
        else total += n; // seconds, or a bare number
    }
    return matched && total > 0 ? Math.round(total) : null;
}

async function ensureNotificationPermission(): Promise<boolean> {
    if (typeof Notification === "undefined") return false;
    if (Notification.permission === "granted") return true;
    if (Notification.permission === "denied") return false;
    try {
        return (await Notification.requestPermission()) === "granted";
    } catch {
        return false;
    }
}

function fire(title: string, body: string): void {
    try {
        if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            new Notification(title, { body });
        }
    } catch { /* */ }
    try {
        navigator.vibrate?.([200, 100, 200]);
    } catch { /* */ }
}

export const timerTool: ToolDef = {
    names: ["set_timer", "timer", "start_timer"],
    async run(_name, args): Promise<ToolRunResult> {
        const durStr =
            typeof args.duration === "string" ? args.duration
                : typeof args.duration === "number" ? String(args.duration)
                    : typeof args.time === "string" ? args.time : "";
        const sec = parseDurationSec(durStr);
        if (sec == null) {
            return { ok: false, summary: `Couldn't read the timer duration "${durStr}". Ask the user for a clearer relative time (e.g. "5 minutes").` };
        }
        const granted = await ensureNotificationPermission();
        setTimeout(() => fire("⏰ Timer done", `Your ${durStr} timer is up.`), sec * 1000);
        const note = granted
            ? "I'll notify you when it's up"
            : "(allow notifications to be alerted; it only fires while this tab stays open)";
        return { ok: true, summary: `Timer set for ${durStr} (${sec}s). ${note}.`, data: { seconds: sec } };
    },
};

export const reminderTool: ToolDef = {
    names: ["set_reminder", "reminder", "remind_me"],
    async run(_name, args): Promise<ToolRunResult> {
        const text = typeof args.text === "string" ? args.text.trim()
            : typeof args.message === "string" ? args.message.trim() : "";
        const timeStr = typeof args.time === "string" ? args.time.trim()
            : typeof args.duration === "string" ? args.duration.trim() : "";
        if (!text) return { ok: false, summary: "No reminder text was given." };
        const sec = parseDurationSec(timeStr);
        if (sec == null) {
            return { ok: false, summary: `Reminders support a relative time only (e.g. "in 10 minutes"); "${timeStr}" wasn't one. Ask the user for a duration.` };
        }
        const granted = await ensureNotificationPermission();
        setTimeout(() => fire("🔔 Reminder", text), sec * 1000);
        const note = granted
            ? "I'll notify you"
            : "(allow notifications to be alerted; it only fires while this tab stays open)";
        return { ok: true, summary: `Reminder set: "${text}" in ${timeStr} (${sec}s). ${note}.`, data: { seconds: sec, text } };
    },
};

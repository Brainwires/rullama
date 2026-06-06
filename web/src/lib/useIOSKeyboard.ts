// useIOSKeyboard — detect when the on-screen keyboard is visible on iOS
// and put the viewport back where it belongs after it dismisses.
//
// Adapted from brainwires-studio/hooks/use-ios-keyboard.ts.
//
// Why we need this even with `html { position: fixed; height: 100dvh }`:
//   On iOS Safari, focusing an <input> near the bottom of the viewport
//   makes the OS scroll the *visual* viewport to keep the input above
//   the keyboard. When the keyboard dismisses, the visual viewport is
//   often left a few px above the layout viewport — the page looks
//   "shifted up" and the safe-area no longer matches the chrome.
//
// The fix is `window.scrollTo(0, 0)` after the keyboard dismisses. The
// hook also exposes the boolean so layouts can shrink themselves while
// the keyboard is up if they need to.

import { useEffect, useLayoutEffect, useState } from "react";

let maxHeight = 0;

function isIOS(): boolean {
    if (typeof navigator === "undefined") return false;
    const ua = navigator.userAgent;
    type MSWindow = Window & { MSStream?: unknown };
    return /iPad|iPhone|iPod/.test(ua) && !(window as MSWindow).MSStream;
}

function keyboardLikelyVisible(): boolean {
    const vv = window.visualViewport;
    if (!vv) return false;
    // Strong signal: the visual viewport is offset upward AND its height
    // plus offset still sums to the historical max — i.e. the keyboard
    // is occupying the bottom slice of the screen.
    if (vv.offsetTop > 0 && vv.height + vv.offsetTop === maxHeight && maxHeight > 0) {
        return true;
    }
    // Fallback heuristic: the visual viewport is meaningfully smaller
    // than the layout viewport. 200 px threshold is wide enough to
    // distinguish a keyboard from the URL-bar resize that happens during
    // normal scroll on iOS Safari.
    return vv.height < window.innerHeight - 200;
}

/**
 * Detects iOS on-screen keyboard visibility and (optionally) snaps the
 * window back to the top once it dismisses.
 *
 * @param scrollFix — when true, calls `window.scrollTo(0, 0)` on keyboard
 *                    dismiss. Defaults to true: that's the iOS-PWA UX you
 *                    nearly always want.
 */
export function useIOSKeyboard(scrollFix: boolean = true): boolean {
    const [visible, setVisible] = useState(false);

    useLayoutEffect(() => {
        if (!isIOS() || !window.visualViewport) return;
        const vv = window.visualViewport;
        const abort = new AbortController();
        let pendingShow: ReturnType<typeof setTimeout> | null = null;

        const handle = () => {
            if (vv.height > maxHeight) maxHeight = vv.height;
            const kbd = keyboardLikelyVisible();
            if (kbd) {
                if (pendingShow) clearTimeout(pendingShow);
                pendingShow = setTimeout(() => setVisible(true), 100);
            } else {
                if (pendingShow) { clearTimeout(pendingShow); pendingShow = null; }
                setVisible(false);
            }
        };

        vv.addEventListener("resize", handle, { signal: abort.signal });
        window.addEventListener("resize", handle, { signal: abort.signal });
        handle();

        if (scrollFix) {
            // Initial snap — covers the case where the page loads with the
            // page already shifted (rare, but happens after a hot reload).
            setTimeout(() => window.scrollTo({ top: 0, behavior: "auto" }), 200);
        }

        return () => {
            abort.abort();
            if (pendingShow) clearTimeout(pendingShow);
        };
    }, [scrollFix]);

    useEffect(() => {
        if (scrollFix && !visible) {
            // Snap back on dismiss. `auto` (not smooth) — we don't want
            // the user to see the page slide.
            window.scrollTo({ top: 0, behavior: "auto" });
        }
    }, [visible, scrollFix]);

    return visible;
}

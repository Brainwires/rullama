import {
    useEffect,
    useRef,
    useState,
    type FC,
    type ReactNode,
    type TouchEvent,
} from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Slimmed adaptation of brainwires-studio's DualSidebarLayout. Same
// pattern: top header (3rem) + body of `calc(100dvh - 3rem)` split into
// left sidebar / main / right sidebar. Desktop sidebars share screen
// (main shrinks); mobile sidebars overlay as drawers with backdrop.
// Swipe-to-close on mobile with vertical-drift + velocity guards so the
// gesture doesn't fight chat-history scrolling.

const HEADER_HEIGHT_PX  = 48;   // 3rem
const TRANSITION_MS     = 220;
const MIN_SWIPE_VELOCITY = 0.3; // px / ms
const MAX_VERTICAL_DRIFT = 100; // px — exceed → assume scroll, not swipe

interface Props {
    children:               ReactNode;
    leftSidebar?:           ReactNode;
    rightSidebar?:          ReactNode;
    leftOpen?:              boolean;
    rightOpen?:             boolean;
    onToggleLeft?:          (next: boolean) => void;
    onToggleRight?:         (next: boolean) => void;
    leftWidth?:             number;
    rightWidth?:            number;
    swipeThreshold?:        number;
    /** Forces the chevron toggle buttons to render even when a sidebar
     *  is empty (default: only renders when content is present). */
    hideToggles?:           boolean;
}

export const DualSidebarLayout: FC<Props> = ({
    children,
    leftSidebar,
    rightSidebar,
    leftOpen = false,
    rightOpen = false,
    onToggleLeft,
    onToggleRight,
    leftWidth = 280,
    rightWidth = 320,
    swipeThreshold = 60,
    hideToggles = false,
}) => {
    // Sidebar visibility — controlled by the parent (App owns persistence).
    const isLeftOpen  = leftOpen;
    const isRightOpen = rightOpen;
    const toggleLeft  = () => onToggleLeft?.(!isLeftOpen);
    const toggleRight = () => onToggleRight?.(!isRightOpen);

    // Mobile vs desktop. `null` while we're still measuring (avoids SSR
    // hydration mismatches and stops a flash of the wrong layout).
    const [isMobile, setIsMobile] = useState<boolean | null>(null);
    useEffect(() => {
        const check = () => setIsMobile(window.innerWidth < 768);
        check();
        window.addEventListener("resize", check);
        return () => window.removeEventListener("resize", check);
    }, []);

    // Swipe-to-close. Velocity + vertical-drift filters keep this from
    // hijacking vertical scrolls inside the sidebar content.
    const leftSwipe  = useRef<{ x: number; y: number; t: number } | null>(null);
    const rightSwipe = useRef<{ x: number; y: number; t: number } | null>(null);

    const onLeftTouchStart = (e: TouchEvent) => {
        if (!isLeftOpen) return;
        const t = e.touches[0];
        leftSwipe.current = { x: t.clientX, y: t.clientY, t: Date.now() };
    };
    const onLeftTouchMove = (e: TouchEvent) => {
        if (!isLeftOpen || !leftSwipe.current) return;
        const t = e.touches[0];
        const dx = t.clientX - leftSwipe.current.x;
        const dy = Math.abs(t.clientY - leftSwipe.current.y);
        if (dy > MAX_VERTICAL_DRIFT) { leftSwipe.current = null; return; }
        const dt = Date.now() - leftSwipe.current.t;
        if (dx < -swipeThreshold && dt > 0 && Math.abs(dx) / dt >= MIN_SWIPE_VELOCITY) {
            toggleLeft();
            leftSwipe.current = null;
        }
    };
    const onLeftTouchEnd = () => { leftSwipe.current = null; };

    const onRightTouchStart = (e: TouchEvent) => {
        if (!isRightOpen) return;
        const t = e.touches[0];
        rightSwipe.current = { x: t.clientX, y: t.clientY, t: Date.now() };
    };
    const onRightTouchMove = (e: TouchEvent) => {
        if (!isRightOpen || !rightSwipe.current) return;
        const t = e.touches[0];
        const dx = t.clientX - rightSwipe.current.x;
        const dy = Math.abs(t.clientY - rightSwipe.current.y);
        if (dy > MAX_VERTICAL_DRIFT) { rightSwipe.current = null; return; }
        const dt = Date.now() - rightSwipe.current.t;
        if (dx > swipeThreshold && dt > 0 && Math.abs(dx) / dt >= MIN_SWIPE_VELOCITY) {
            toggleRight();
            rightSwipe.current = null;
        }
    };
    const onRightTouchEnd = () => { rightSwipe.current = null; };

    const transition = `${TRANSITION_MS}ms`;

    return (
        <div
            className="relative w-full overflow-hidden bg-background"
            style={{ height: `calc(100dvh - ${HEADER_HEIGHT_PX}px)` }}
        >
            {/* ─── desktop layout ─── */}
            {isMobile === false && (
                <div className="flex size-full overflow-hidden">
                    {leftSidebar && (
                        <aside
                            className="shrink-0 overflow-y-auto overflow-x-hidden border-r border-border bg-card/30 transition-all"
                            style={{ width: isLeftOpen ? leftWidth : 0, transitionDuration: transition }}
                        >
                            <div style={{ width: leftWidth }}>{leftSidebar}</div>
                        </aside>
                    )}
                    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
                        {children}
                        {leftSidebar && !hideToggles && (
                            <ToggleTab side="left"  open={isLeftOpen}  onClick={toggleLeft} />
                        )}
                        {rightSidebar && !hideToggles && (
                            <ToggleTab side="right" open={isRightOpen} onClick={toggleRight} />
                        )}
                    </div>
                    {rightSidebar && (
                        <aside
                            className="shrink-0 overflow-y-auto overflow-x-hidden border-l border-border bg-card/30 transition-all"
                            style={{ width: isRightOpen ? rightWidth : 0, transitionDuration: transition }}
                        >
                            <div style={{ width: rightWidth }}>{rightSidebar}</div>
                        </aside>
                    )}
                </div>
            )}

            {/* ─── mobile layout (overlay drawers) ─── */}
            {isMobile === true && (
                <div className="relative size-full">
                    {/* Backdrop (shared by both sidebars). */}
                    <div
                        className={cn(
                            "fixed inset-x-0 z-30 bg-black/40 transition-opacity",
                            (isLeftOpen || isRightOpen) ? "opacity-100" : "pointer-events-none opacity-0",
                        )}
                        style={{
                            top: HEADER_HEIGHT_PX,
                            bottom: 0,
                            transitionDuration: transition,
                        }}
                        onClick={() => { if (isLeftOpen) toggleLeft(); if (isRightOpen) toggleRight(); }}
                    />

                    {leftSidebar && (
                        <aside
                            className={cn(
                                "fixed left-0 z-40 overflow-y-auto overflow-x-hidden border-r border-border bg-card transition-transform",
                                isLeftOpen ? "translate-x-0" : "-translate-x-full",
                            )}
                            style={{
                                width: leftWidth,
                                top: HEADER_HEIGHT_PX,
                                bottom: 0,
                                transitionDuration: transition,
                            }}
                            onTouchStart={onLeftTouchStart}
                            onTouchMove={onLeftTouchMove}
                            onTouchEnd={onLeftTouchEnd}
                        >
                            {leftSidebar}
                        </aside>
                    )}

                    {rightSidebar && (
                        <aside
                            className={cn(
                                "fixed right-0 z-40 overflow-y-auto overflow-x-hidden border-l border-border bg-card transition-transform",
                                isRightOpen ? "translate-x-0" : "translate-x-full",
                            )}
                            style={{
                                width: rightWidth,
                                top: HEADER_HEIGHT_PX,
                                bottom: 0,
                                transitionDuration: transition,
                            }}
                            onTouchStart={onRightTouchStart}
                            onTouchMove={onRightTouchMove}
                            onTouchEnd={onRightTouchEnd}
                        >
                            {rightSidebar}
                        </aside>
                    )}

                    {/* Main content */}
                    <div className="relative flex size-full min-h-0 flex-col overflow-hidden">
                        {children}
                        {leftSidebar && !hideToggles && (
                            <ToggleTab side="left"  open={isLeftOpen}  onClick={toggleLeft} />
                        )}
                        {rightSidebar && !hideToggles && (
                            <ToggleTab side="right" open={isRightOpen} onClick={toggleRight} />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// Chevron tab pinned to the screen edge — clickable hint that there's a
// sidebar over there. The icon flips to show open/close direction.
function ToggleTab({ side, open, onClick }: { side: "left" | "right"; open: boolean; onClick: () => void }) {
    const Icon = side === "left"
        ? (open ? ChevronLeft : ChevronRight)
        : (open ? ChevronRight : ChevronLeft);
    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={onClick}
            aria-label={`${open ? "Close" : "Open"} ${side} sidebar`}
            className={cn(
                "absolute top-1/2 z-20 h-12 w-5 -translate-y-1/2 rounded-none bg-muted/30 px-0 text-muted-foreground hover:bg-muted/60",
                side === "left" ? "left-0 rounded-r-md" : "right-0 rounded-l-md",
            )}
        >
            <Icon className="h-4 w-4" />
        </Button>
    );
}

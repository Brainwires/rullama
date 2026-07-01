// **Promise-based confirm() API backed by shadcn AlertDialog.**
//
// Replaces native `window.confirm()`. Drop-in: same Promise-returning
// shape, but renders as a React modal we can drive from automation
// (the OK button is a real DOM <button> Playwright/CDP can click) AND
// looks like the rest of the app.
//
// Usage:
//   1. Wrap the app root with `<ConfirmProvider>`:
//        <ConfirmProvider><App/></ConfirmProvider>
//   2. Anywhere inside the tree, get the imperative function:
//        const confirm = useConfirm();
//        if (!(await confirm({title:"Delete X?", description:"…"}))) return;
//
// The hook returns `(opts) => Promise<boolean>`. Resolves true on OK,
// false on Cancel (or Esc / overlay click — Radix wires those to
// onOpenChange(false)).

import * as React from "react";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export interface ConfirmOptions {
    /** Modal title (bold, top of card). */
    title: string;
    /** Body text. Newlines are preserved (whitespace-pre-line). */
    description?: string;
    /** Button text. Default "OK". */
    okLabel?: string;
    /** Button text. Default "Cancel". */
    cancelLabel?: string;
    /** Render the OK button in destructive style. */
    destructive?: boolean;
}

type Resolver = (ok: boolean) => void;

interface ConfirmCtx {
    confirm: (opts: ConfirmOptions) => Promise<boolean>;
}

const Ctx = React.createContext<ConfirmCtx | null>(null);

export function ConfirmProvider({ children }: { children: React.ReactNode }) {
    const [state, setState] = React.useState<{
        opts: ConfirmOptions;
        resolver: Resolver;
    } | null>(null);

    const confirm = React.useCallback(
        (opts: ConfirmOptions) =>
            new Promise<boolean>((resolve) => {
                setState({ opts, resolver: resolve });
            }),
        [],
    );

    const handleOk = () => {
        state?.resolver(true);
        setState(null);
    };
    const handleCancel = () => {
        state?.resolver(false);
        setState(null);
    };

    return (
        <Ctx.Provider value={{ confirm }}>
            {children}
            <AlertDialog
                open={!!state}
                onOpenChange={(open) => {
                    if (!open) handleCancel();
                }}
            >
                <AlertDialogContent data-testid="confirm-dialog">
                    <AlertDialogHeader>
                        <AlertDialogTitle>{state?.opts.title}</AlertDialogTitle>
                        {state?.opts.description ? (
                            <AlertDialogDescription>
                                {state.opts.description}
                            </AlertDialogDescription>
                        ) : null}
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel onClick={handleCancel} data-testid="confirm-cancel">
                            {state?.opts.cancelLabel ?? "Cancel"}
                        </AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleOk}
                            data-testid="confirm-ok"
                            className={
                                state?.opts.destructive
                                    ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                    : undefined
                            }
                        >
                            {state?.opts.okLabel ?? "OK"}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </Ctx.Provider>
    );
}

/**
 * Imperative confirm. Throws if no `<ConfirmProvider>` is mounted —
 * fail-fast is better than silently falling back to window.confirm
 * (which is what we're explicitly replacing).
 */
export function useConfirm(): ConfirmCtx["confirm"] {
    const ctx = React.useContext(Ctx);
    if (!ctx) {
        throw new Error("useConfirm() called outside <ConfirmProvider>");
    }
    return ctx.confirm;
}

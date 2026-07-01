import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
    "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
    {
        variants: {
            tone: {
                ok:    "bg-green-500/15 text-green-400 ring-green-500/30",
                warn:  "bg-yellow-500/15 text-yellow-400 ring-yellow-500/30",
                err:   "bg-destructive/15 text-destructive ring-destructive/30",
                muted: "bg-muted text-muted-foreground ring-border",
                info:  "bg-primary/15 text-primary ring-primary/30",
            },
        },
        defaultVariants: { tone: "muted" },
    },
);

export interface BadgeProps
    extends React.HTMLAttributes<HTMLSpanElement>,
            VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
    return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

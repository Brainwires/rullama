/** @type {import('tailwindcss').Config} */
export default {
    darkMode: ["class"],
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        container: {
            center: true,
            padding: "1rem",
            screens: { "2xl": "1320px" },
        },
        extend: {
            colors: {
                border:      "hsl(var(--border))",
                input:       "hsl(var(--input))",
                ring:        "hsl(var(--ring))",
                background:  "hsl(var(--background))",
                foreground:  "hsl(var(--foreground))",
                primary: {
                    DEFAULT:  "hsl(var(--primary))",
                    foreground:"hsl(var(--primary-foreground))",
                },
                muted: {
                    DEFAULT:  "hsl(var(--muted))",
                    foreground:"hsl(var(--muted-foreground))",
                },
                accent: {
                    DEFAULT:  "hsl(var(--accent))",
                    foreground:"hsl(var(--accent-foreground))",
                },
                destructive: {
                    DEFAULT:  "hsl(var(--destructive))",
                    foreground:"hsl(var(--destructive-foreground))",
                },
                card: {
                    DEFAULT:  "hsl(var(--card))",
                    foreground:"hsl(var(--card-foreground))",
                },
            },
            borderRadius: {
                lg: "var(--radius)",
                md: "calc(var(--radius) - 2px)",
                sm: "calc(var(--radius) - 4px)",
            },
            keyframes: {
                "fade-in":  { from: { opacity: "0" }, to: { opacity: "1" } },
                "slide-up": { from: { transform: "translateY(8px)", opacity: "0" },
                              to:   { transform: "translateY(0)",   opacity: "1" } },
            },
            animation: {
                "fade-in":  "fade-in 120ms ease-out",
                "slide-up": "slide-up 180ms ease-out",
            },
        },
    },
    plugins: [require("tailwindcss-animate")],
};

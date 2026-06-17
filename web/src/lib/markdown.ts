// Markdown → sanitized HTML. Adapted from brainwires-chat-pwa/src/markdown.js.
//
// `marked` produces HTML; DOMPurify scrubs anything not in our allowlist.
// A custom code renderer wraps every fenced block in `<div class="codeblock">`
// with an inline copy button (delegated handler is attached in MessageBubble).
// Cheap enough to call on every streaming chunk.

import DOMPurify from "dompurify";
import { marked, Renderer, type Tokens } from "marked";
import markedKatex from "marked-katex-extension";

function escapeHtml(s: string): string {
    return s
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

const renderer = new Renderer();

renderer.code = function code({ text, lang }: Tokens.Code): string {
    const langStr  = (lang || "").trim();
    const langCls  = langStr ? ` class="language-${escapeHtml(langStr)}"` : "";
    const escaped  = escapeHtml(text);
    return `<div class="codeblock">`
         + `<button type="button" class="codeblock-copy" data-bw-copy="1">copy</button>`
         + `<pre><code${langCls}>${escaped}</code></pre>`
         + `</div>`;
};

renderer.link = function link({ href, title, tokens }: Tokens.Link): string {
    const text  = this.parser.parseInline(tokens);
    const t     = title ? ` title="${escapeHtml(title)}"` : "";
    return `<a href="${escapeHtml(href ?? "")}"${t} target="_blank" rel="noopener noreferrer">${text}</a>`;
};

marked.setOptions({ gfm: true, breaks: true, pedantic: false, renderer });

// LaTeX math via KaTeX. `$…$` inline / `$$…$$` block (the markers models
// reach for). `nonStandard: true` accepts `$x$` even with adjacent
// whitespace (models aren't strict); `throwOnError: false` renders malformed
// TeX as-is instead of blowing up a whole reply. `output: "html"` skips the
// MathML twin so there's nothing extra for DOMPurify to second-guess —
// katex emits plain <span>s with class/style/aria-hidden, all kept below.
marked.use(markedKatex({ throwOnError: false, nonStandard: true, output: "html" }));

const PURIFY_CONFIG = {
    ADD_ATTR: ["data-bw-copy", "target", "aria-hidden"] as string[],
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel|data:image\/[a-z+.\-]+;base64,):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
};

let _purify: typeof DOMPurify | null = null;
function getPurify(): typeof DOMPurify | null {
    if (_purify) return _purify;
    if (typeof window === "undefined" || typeof document === "undefined") return null;
    _purify = DOMPurify;
    return _purify;
}

/** Render markdown to sanitized HTML. */
export function renderMarkdown(src: string): string {
    if (!src) return "";
    const html = marked.parse(src, { async: false }) as string;
    const p = getPurify();
    if (!p) return html;
    return p.sanitize(html, PURIFY_CONFIG) as unknown as string;
}

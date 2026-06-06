// Canvas-based image preprocessor for the Gemma 4 vision tower.
// Ported from image_preprocess.js.
//
// Mirrors Ollama's process_image.go::ProcessImage:
//   * smartResize aligning to (patch_size × n_merge) = 48 px boundaries
//     while preserving aspect ratio and capping pixel count to MAX_PIXELS
//     (so the post-pool soft-token count stays under the model's max).
//   * Normalise (pixel/255) * 2 - 1 → range [-1, 1].
//   * Output channel-first Float32Array: [R..., G..., B...] of length 3*H*W.

const ALIGN        = 48;           // patch_size (16) × n_merge (3)
const MIN_TOKENS   = 40;
const MAX_TOKENS   = 280;
const PATCH_AREA   = 16 * 16 * 3 * 3;
// const MIN_PIXELS = MIN_TOKENS * PATCH_AREA;
const MAX_PIXELS   = MAX_TOKENS * PATCH_AREA;
// Matches Rust's `vision::MAX_IMG_DIM`. The pixel budget bounds total area,
// not per-dim, so an extreme aspect ratio could land beyond this even when
// MAX_PIXELS is respected. Clamp here to stay inside the Rust scratch-buffer
// and aligned to ALIGN.
const MAX_DIM      = 1536;

export interface ProcessedImage {
    pixels:  Float32Array;
    h:       number;
    w:       number;
    /** Small JPEG thumbnail (≤ 192 px on the long edge) for chat
     *  bubble + input-row preview, separate from the model-input
     *  pixels which are channel-first [-1, 1] floats. */
    dataUrl: string;
}

function smartResize(origW: number, origH: number): { targetW: number; targetH: number } {
    const totalPx = origW * origH;
    let targetW: number, targetH: number;
    if (MAX_PIXELS > 0 && totalPx > 0) {
        const factor = Math.sqrt(MAX_PIXELS / totalPx);
        targetH = Math.max(ALIGN, Math.floor(factor * origH / ALIGN) * ALIGN);
        targetW = Math.max(ALIGN, Math.floor(factor * origW / ALIGN) * ALIGN);
    } else {
        targetH = Math.max(ALIGN, Math.floor(origH / ALIGN) * ALIGN);
        targetW = Math.max(ALIGN, Math.floor(origW / ALIGN) * ALIGN);
    }
    // Per-dim cap (MAX_DIM) for extreme aspect ratios — even when MAX_PIXELS
    // is satisfied, a very long-and-thin image could exceed the Rust
    // scratch-buffer's per-axis size. Clamp and re-align to ALIGN.
    if (targetW > MAX_DIM) targetW = Math.floor(MAX_DIM / ALIGN) * ALIGN;
    if (targetH > MAX_DIM) targetH = Math.floor(MAX_DIM / ALIGN) * ALIGN;
    // MIN_TOKENS floor (not enforced in legacy preprocessor either,
    // but here for completeness — small images get padded by the
    // resize to a minimum size).
    void MIN_TOKENS;
    void PATCH_AREA;
    return { targetW, targetH };
}

async function decodeImage(blob: Blob): Promise<HTMLImageElement> {
    const url = URL.createObjectURL(blob);
    try {
        const img = new Image();
        await new Promise<void>((resolve, reject) => {
            img.onload  = () => resolve();
            img.onerror = (e) => reject(new Error(`image decode failed: ${String(e)}`));
            img.src     = url;
        });
        return img;
    } finally {
        URL.revokeObjectURL(url);
    }
}

/** Process a Blob (PNG/JPEG/etc.) into model-ready pixel data + a small
 *  thumbnail for the UI. */
export async function preprocessImage(blob: Blob): Promise<ProcessedImage> {
    const img = await decodeImage(blob);
    const { targetW, targetH } = smartResize(img.naturalWidth, img.naturalHeight);

    const canvas = document.createElement("canvas");
    canvas.width  = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, targetW, targetH);
    const imgData = ctx.getImageData(0, 0, targetW, targetH).data;

    // RGBA → channel-first [R..., G..., B...] f32 in [-1, 1].
    const n = targetW * targetH;
    const pixels = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        const o = i * 4;
        pixels[i]         = (imgData[o    ] / 255) * 2 - 1;
        pixels[i + n]     = (imgData[o + 1] / 255) * 2 - 1;
        pixels[i + 2 * n] = (imgData[o + 2] / 255) * 2 - 1;
    }

    // Thumbnail for the chat bubble + input-row preview.
    const thumbW = 192;
    const thumbScale = Math.min(1, thumbW / Math.max(targetW, targetH));
    const thumbCanvas = document.createElement("canvas");
    thumbCanvas.width  = Math.max(1, Math.round(targetW * thumbScale));
    thumbCanvas.height = Math.max(1, Math.round(targetH * thumbScale));
    const tctx = thumbCanvas.getContext("2d");
    if (!tctx) throw new Error("Canvas 2D context unavailable (thumb)");
    tctx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const dataUrl = thumbCanvas.toDataURL("image/jpeg", 0.85);

    return { pixels, h: targetH, w: targetW, dataUrl };
}

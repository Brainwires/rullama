// Canvas-based image preprocessor for the Gemma 4 vision tower.
//
// Mirrors Ollama's process_image.go::ProcessImage:
//   * smartResize aligning to (patch_size × n_merge) = 48 px boundaries while
//     preserving aspect ratio and capping pixel count to maxPixels (so the
//     post-pool soft-token count stays under the model's max).
//   * Normalise (pixel/255) * 2 - 1 → range [-1, 1].
//   * Output channel-first Float32Array: [R..., G..., B...] of length 3*H*W.

const ALIGN = 48;          // patch_size (16) × n_merge (3) — must come from Rust if those change
const MIN_TOKENS = 40;
const MAX_TOKENS = 280;

// Each output token covers patchSize² × nMerge² = 16² × 3² = 2304 pixels.
const PATCH_AREA = 16 * 16 * 3 * 3;
const MIN_PIXELS = MIN_TOKENS * PATCH_AREA;
const MAX_PIXELS = MAX_TOKENS * PATCH_AREA;

/** smartResize — preserve aspect ratio, scale to fill maxPixels, snap to 48 px. */
function smartResize(origW, origH) {
    const totalPx = origW * origH;
    let targetW, targetH;
    if (MAX_PIXELS > 0 && totalPx > 0) {
        const factor = Math.sqrt(MAX_PIXELS / totalPx);
        targetH = Math.max(ALIGN, Math.floor(factor * origH / ALIGN) * ALIGN);
        targetW = Math.max(ALIGN, Math.floor(factor * origW / ALIGN) * ALIGN);
    } else {
        targetH = Math.max(ALIGN, Math.floor(origH / ALIGN) * ALIGN);
        targetW = Math.max(ALIGN, Math.floor(origW / ALIGN) * ALIGN);
    }
    return { targetW, targetH };
}

/** Decode a Blob (image file) into an HTMLImageElement once it's loaded. */
async function decodeImage(blob) {
    const url = URL.createObjectURL(blob);
    try {
        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload  = () => resolve();
            img.onerror = (e) => reject(new Error(`image decode: ${e}`));
            img.src = url;
        });
        return img;
    } finally {
        URL.revokeObjectURL(url);
    }
}

/** Process a Blob (PNG/JPEG/etc) into the format `Model.encodeImage` wants.
 *  Returns `{pixels: Float32Array, h, w, dataUrl}`. `dataUrl` is a small
 *  thumbnail (≤ 256 px on the long edge) for the chat bubble — distinct from
 *  the model-input pixels.
 */
export async function preprocessImage(blob) {
    const img = await decodeImage(blob);
    const { targetW, targetH } = smartResize(img.naturalWidth, img.naturalHeight);

    // Draw the resized image onto a canvas, read RGBA pixel data.
    const canvas = document.createElement("canvas");
    canvas.width  = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    // Default smoothing is bilinear-ish, matching Ollama's draw.BiLinear.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, targetW, targetH);
    const imgData = ctx.getImageData(0, 0, targetW, targetH).data;

    // Repack RGBA → channel-first [R..., G..., B...] f32 in [-1, 1].
    const n = targetW * targetH;
    const pixels = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        const o = i * 4;
        pixels[i]         = (imgData[o    ] / 255) * 2 - 1; // R
        pixels[i + n]     = (imgData[o + 1] / 255) * 2 - 1; // G
        pixels[i + 2 * n] = (imgData[o + 2] / 255) * 2 - 1; // B
    }

    // Thumbnail for the chat bubble: small canvas, JPEG encode.
    const thumbW = 192;
    const thumbScale = Math.min(1, thumbW / Math.max(targetW, targetH));
    const thumbCanvas = document.createElement("canvas");
    thumbCanvas.width  = Math.max(1, Math.round(targetW * thumbScale));
    thumbCanvas.height = Math.max(1, Math.round(targetH * thumbScale));
    const tctx = thumbCanvas.getContext("2d");
    tctx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const dataUrl = thumbCanvas.toDataURL("image/jpeg", 0.85);

    return { pixels, h: targetH, w: targetW, dataUrl };
}

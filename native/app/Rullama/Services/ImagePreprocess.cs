using System;
using SkiaSharp;

namespace Rullama.Services;

public readonly record struct ProcessedImage(float[] Pixels, int Height, int Width);

/// <summary>
/// Gemma 4 vision-tower preprocessing (port of web/src/lib/image_preprocess.ts,
/// mirroring Ollama's process_image.go): smart-resize aligned to 48px with an
/// area cap, normalise (px/255)*2-1, output channel-first f32 [R…,G…,B…].
/// </summary>
public static class ImagePreprocess
{
    private const int Align = 48;          // patch_size(16) × n_merge(3)
    private const int MaxTokens = 280;
    private const int PatchArea = 16 * 16 * 3 * 3;
    private const int MaxPixels = MaxTokens * PatchArea;
    private const int MaxDim = 1536;

    public static ProcessedImage Process(byte[] imageBytes)
    {
        using SKBitmap? decoded = SKBitmap.Decode(imageBytes)
            ?? throw new InvalidOperationException("Could not decode image.");

        (int targetW, int targetH) = SmartResize(decoded.Width, decoded.Height);

        var info = new SKImageInfo(targetW, targetH, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using SKBitmap resized = decoded.Resize(info, SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Image resize failed.");

        ReadOnlySpan<byte> rgba = resized.GetPixelSpan();
        int n = targetW * targetH;
        var pixels = new float[3 * n];
        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            pixels[i] = rgba[o] / 255f * 2f - 1f;
            pixels[i + n] = rgba[o + 1] / 255f * 2f - 1f;
            pixels[i + 2 * n] = rgba[o + 2] / 255f * 2f - 1f;
        }
        return new ProcessedImage(pixels, targetH, targetW);
    }

    private static (int W, int H) SmartResize(int origW, int origH)
    {
        long total = (long)origW * origH;
        int targetW, targetH;
        if (total > 0)
        {
            double factor = Math.Sqrt((double)MaxPixels / total);
            targetH = Math.Max(Align, (int)(factor * origH / Align) * Align);
            targetW = Math.Max(Align, (int)(factor * origW / Align) * Align);
        }
        else
        {
            targetW = targetH = Align;
        }
        if (targetW > MaxDim) targetW = MaxDim / Align * Align;
        if (targetH > MaxDim) targetH = MaxDim / Align * Align;
        return (targetW, targetH);
    }
}

using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>Result of a generation: RGBA8 pixels + dimensions.</summary>
public readonly record struct GeneratedImageData(byte[] Rgba, int Width, int Height);

/// <summary>
/// Managed wrapper over the rust-core image-generation handle (M13), which drives
/// the native <c>imagegen::ImageBundle</c> pipeline (Qwen3 → DiT → VAE). Mirrors
/// the other engine wrappers (TTS/clone/embed). Blocking calls run off the UI thread.
/// </summary>
public sealed class RustImageGen : IDisposable
{
    private IntPtr _handle;

    public RustImageGen()
    {
        _handle = RustCore.rl_imagegen_create();
        if (_handle == IntPtr.Zero) throw new InvalidOperationException("rl_imagegen_create failed");
    }

    /// <summary>Load a Z-Image model directory (text_encoder/ transformer/ vae/ tokenizer/).</summary>
    public void LoadBlobs(string dir)
    {
        int rc = RustCore.rl_imagegen_load_blobs(_handle, dir);
        if (rc != 0) throw new InvalidOperationException($"image model load failed: {RustCore.LastError()}");
    }

    /// <summary>Generate an image. <paramref name="latent"/> is the latent size in
    /// cells (image pixels = latent × VAE downscale, typically 8×). Returns RGBA8
    /// pixels + dimensions; reports progress per encode/denoise/VAE step.</summary>
    public GeneratedImageData Generate(
        string prompt, string negPrompt, float cfgScale, uint latent, uint steps, ulong seed,
        Action<uint, uint, string>? onProgress)
    {
        RustCore.ImageProgressCallback cb = (_, step, total, stagePtr) =>
        {
            string stage = stagePtr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(stagePtr) ?? string.Empty;
            onProgress?.Invoke(step, total, stage);
        };
        int rc = RustCore.rl_imagegen_generate(
            _handle, prompt, negPrompt ?? string.Empty, cfgScale, latent, latent, steps, seed,
            cb, IntPtr.Zero, out IntPtr ptr, out UIntPtr len, out uint w, out uint h);
        GC.KeepAlive(cb);
        if (rc != 0) throw new InvalidOperationException($"image generation failed: {RustCore.LastError()}");
        try
        {
            var bytes = new byte[(int)len];
            if (bytes.Length > 0) Marshal.Copy(ptr, bytes, 0, bytes.Length);
            return new GeneratedImageData(bytes, (int)w, (int)h);
        }
        finally { RustCore.rl_free_bytes(ptr, len); }
    }

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            RustCore.rl_imagegen_free(_handle);
            _handle = IntPtr.Zero;
        }
    }
}

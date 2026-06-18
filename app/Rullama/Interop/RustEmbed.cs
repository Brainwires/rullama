using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>Managed wrapper over a rust-core EmbeddingGemma handle (own thread).</summary>
internal sealed class RustEmbed : IDisposable
{
    private IntPtr _h;

    public RustEmbed()
    {
        _h = RustCore.rl_embed_create();
        if (_h == IntPtr.Zero)
            throw new InvalidOperationException($"rl_embed_create failed: {RustCore.LastError()}");
    }

    public void LoadPath(string path)
    {
        if (RustCore.rl_embed_load_path(_h, path) != 0)
            throw new InvalidOperationException($"embed load failed: {RustCore.LastError()}");
    }

    public int Dim => (int)RustCore.rl_embed_dim(_h);

    /// <summary>Embed text → unit-comparable vector of length <paramref name="targetDim"/>.</summary>
    public float[] Embed(string text, int targetDim)
    {
        if (RustCore.rl_embed(_h, text, (UIntPtr)targetDim, out IntPtr p, out UIntPtr n) != 0)
            throw new InvalidOperationException($"embed failed: {RustCore.LastError()}");
        try
        {
            int count = (int)n;
            var v = new float[count];
            if (count > 0) Marshal.Copy(p, v, 0, count);
            return v;
        }
        finally { RustCore.rl_free_f32(p, n); }
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { RustCore.rl_embed_free(_h); _h = IntPtr.Zero; }
    }
}

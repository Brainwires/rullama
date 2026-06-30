using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>
/// Managed wrapper over a rust-core Kokoro TTS handle (its own dedicated native
/// thread). Blocking calls (Load, Synthesize) should run off the UI thread.
/// </summary>
internal sealed class RustTts : IDisposable
{
    private IntPtr _h;

    public RustTts()
    {
        _h = RustCore.rl_tts_create();
        if (_h == IntPtr.Zero)
            throw new InvalidOperationException($"rl_tts_create failed: {RustCore.LastError()}");
    }

    private void Check(int rc, string op)
    {
        if (rc != 0)
            throw new InvalidOperationException($"{op} failed (rc={rc}): {RustCore.LastError()}");
    }

    public void LoadPath(string path) => Check(RustCore.rl_tts_load_path(_h, path), "tts load");

    public void SetLexicon(string goldPath, string silverPath)
        => Check(RustCore.rl_tts_set_lexicon(_h, goldPath, silverPath), "tts set_lexicon");

    public int SampleRate => (int)RustCore.rl_tts_sample_rate(_h);

    /// <summary>Synthesize text → mono f32 PCM at <see cref="SampleRate"/>.</summary>
    public float[] Synthesize(string text, string voice)
    {
        Check(RustCore.rl_tts_synthesize(_h, text, voice, out IntPtr p, out UIntPtr n), "tts synthesize");
        try
        {
            int count = (int)n;
            var pcm = new float[count];
            if (count > 0) Marshal.Copy(p, pcm, 0, count);
            return pcm;
        }
        finally { RustCore.rl_free_f32(p, n); }
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero)
        {
            RustCore.rl_tts_free(_h);
            _h = IntPtr.Zero;
        }
    }
}

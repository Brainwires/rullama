using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>Managed wrapper over a rust-core StyleTTS2 voice-clone handle.</summary>
internal sealed class RustClone : IDisposable
{
    private IntPtr _h;

    public RustClone()
    {
        _h = RustCore.rl_clone_create();
        if (_h == IntPtr.Zero)
            throw new InvalidOperationException($"rl_clone_create failed: {RustCore.LastError()}");
    }

    private void Check(int rc, string op)
    {
        if (rc != 0) throw new InvalidOperationException($"{op} failed (rc={rc}): {RustCore.LastError()}");
    }

    public void LoadPath(string path) => Check(RustCore.rl_clone_load_path(_h, path), "clone load");
    public void SetLexicon(string gold, string silver) => Check(RustCore.rl_clone_set_lexicon(_h, gold, silver), "clone set_lexicon");
    public int SampleRate => (int)RustCore.rl_clone_sample_rate(_h);

    /// <summary>Encode a 24 kHz mono reference clip → speaker-voice vector.</summary>
    public float[] EncodeVoice(float[] pcm24k)
    {
        Check(RustCore.rl_clone_encode_voice(_h, pcm24k, (UIntPtr)pcm24k.Length, out IntPtr p, out UIntPtr n), "encode_voice");
        try { return ReadF32(p, n); } finally { RustCore.rl_free_f32(p, n); }
    }

    /// <summary>Synthesize text with a cloned voice vector → 24 kHz PCM.</summary>
    public float[] Synthesize(string text, float[] voice)
    {
        Check(RustCore.rl_clone_synthesize(_h, text, voice, (UIntPtr)voice.Length, out IntPtr p, out UIntPtr n), "clone synthesize");
        try { return ReadF32(p, n); } finally { RustCore.rl_free_f32(p, n); }
    }

    private static float[] ReadF32(IntPtr p, UIntPtr n)
    {
        int count = (int)n;
        var v = new float[count];
        if (count > 0) Marshal.Copy(p, v, 0, count);
        return v;
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { RustCore.rl_clone_free(_h); _h = IntPtr.Zero; }
    }
}

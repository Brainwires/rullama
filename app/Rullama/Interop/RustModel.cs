using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>
/// Managed wrapper over a rust-core engine handle. One instance == one loaded
/// model on one dedicated native thread. All blocking calls (Load, Generate)
/// should be invoked off the UI thread; <see cref="Cancel"/> is thread-safe.
/// </summary>
internal sealed class RustModel : IDisposable
{
    private IntPtr _h;

    public RustModel()
    {
        _h = RustCore.rl_model_create();
        if (_h == IntPtr.Zero)
            throw new InvalidOperationException($"rl_model_create failed: {RustCore.LastError()}");
    }

    private void Check(int rc, string op)
    {
        if (rc != 0)
            throw new InvalidOperationException($"{op} failed (rc={rc}): {RustCore.LastError()}");
    }

    /// <summary>Load a GGUF by path. Blocking — call off the UI thread.</summary>
    public void LoadPath(string path, uint maxContext = 0, bool textOnly = false)
        => Check(RustCore.rl_model_load_path(_h, path, maxContext, textOnly ? 1 : 0), "load");

    public uint[] Encode(string text)
    {
        Check(RustCore.rl_encode(_h, text, out IntPtr ids, out UIntPtr n), "encode");
        int count = (int)n;
        var arr = new uint[count];
        if (count > 0)
        {
            var tmp = new int[count];
            Marshal.Copy(ids, tmp, 0, count);
            Buffer.BlockCopy(tmp, 0, arr, 0, count * sizeof(uint));
        }
        RustCore.rl_free_u32(ids, n);
        return arr;
    }

    /// <summary>Raw vocab string for a token id, or null if unmapped.</summary>
    public string? TokenStr(uint id)
    {
        if (RustCore.rl_token_str(_h, id, out IntPtr p) != 0)
            return null;
        try { return Marshal.PtrToStringUTF8(p); }
        finally { RustCore.rl_free_str(p); }
    }

    /// <summary>Display text for a token (SentencePiece ▁ → space).</summary>
    public string Decode(uint id) => (TokenStr(id) ?? string.Empty).Replace('▁', ' ');

    public void SetSampling(float temperature, uint topK, float topP, float repetitionPenalty, ulong seed)
        => Check(RustCore.rl_set_sampling(_h, temperature, topK, topP, repetitionPenalty, seed), "set_sampling");

    public void Reset() => Check(RustCore.rl_reset(_h), "reset");

    public uint VocabSize() => RustCore.rl_vocab_size(_h);
    public uint Position() => RustCore.rl_position(_h);

    /// <summary>
    /// Stream a generation. Feeds <paramref name="prompt"/> then decodes up to
    /// <paramref name="maxNew"/> tokens, invoking <paramref name="onPiece"/> per
    /// token with its decoded display text (on the native worker thread — marshal
    /// to the UI thread yourself). Blocking; returns the number of tokens
    /// produced. Use <see cref="Cancel"/> to stop early.
    /// </summary>
    public int Generate(uint[] prompt, uint maxNew, Action<string> onPiece)
    {
        // Keep the delegate rooted for the duration of the (synchronous) call.
        RustCore.TokenCallback cb = (_, _, piece, _) =>
        {
            string s = piece == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(piece) ?? string.Empty;
            onPiece(s);
        };
        int produced = RustCore.rl_generate(_h, prompt, (UIntPtr)prompt.Length, maxNew, cb, IntPtr.Zero);
        GC.KeepAlive(cb);
        if (produced < 0)
            throw new InvalidOperationException($"generate failed (rc={produced}): {RustCore.LastError()}");
        return produced;
    }

    // ---- multimodal ----
    public bool HasVision() => RustCore.rl_has_vision(_h) != 0;
    public bool HasAudio() => RustCore.rl_has_audio(_h) != 0;

    public (uint Begin, uint End)? ImageSentinels()
        => RustCore.rl_image_sentinel_ids(_h, out uint b, out uint e) == 0 ? (b, e) : null;

    public (uint Begin, uint End)? AudioSentinels()
        => RustCore.rl_audio_sentinel_ids(_h, out uint b, out uint e) == 0 ? (b, e) : null;

    public long ImageSoftTokenCount(int h, int w)
        => RustCore.rl_image_soft_token_count(_h, (UIntPtr)h, (UIntPtr)w);

    public float[] EncodeImage(float[] pixels, int h, int w)
    {
        Check(RustCore.rl_encode_image(_h, pixels, (UIntPtr)pixels.Length, (UIntPtr)h, (UIntPtr)w, out IntPtr p, out UIntPtr n), "encode_image");
        try { return ReadF32(p, n); } finally { RustCore.rl_free_f32(p, n); }
    }

    public float[] EncodeAudio(float[] pcm)
    {
        Check(RustCore.rl_encode_audio(_h, pcm, (UIntPtr)pcm.Length, out IntPtr p, out UIntPtr n), "encode_audio");
        try { return ReadF32(p, n); } finally { RustCore.rl_free_f32(p, n); }
    }

    /// <summary>Decode WAV bytes to mono f32 PCM. Standalone (no model needed).</summary>
    public static float[] DecodeWav(byte[] bytes)
    {
        if (RustCore.rl_decode_wav(bytes, (UIntPtr)bytes.Length, out IntPtr p, out UIntPtr n) != 0)
            throw new InvalidOperationException("decode_wav failed: " + RustCore.LastError());
        try { return ReadF32(p, n); } finally { RustCore.rl_free_f32(p, n); }
    }

    /// <summary>Streaming generate with one spliced media item (see rust-core).</summary>
    public int GenerateSpliced(uint[] prompt, uint sentinelBegin, float[] soft, int dText, uint maxNew, Action<string> onPiece)
    {
        RustCore.TokenCallback cb = (_, _, piece, _) =>
        {
            string s = piece == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(piece) ?? string.Empty;
            onPiece(s);
        };
        int produced = RustCore.rl_generate_spliced(
            _h, prompt, (UIntPtr)prompt.Length, sentinelBegin, soft, (UIntPtr)soft.Length, (UIntPtr)dText, maxNew, cb, IntPtr.Zero);
        GC.KeepAlive(cb);
        if (produced < 0)
            throw new InvalidOperationException($"generate_spliced failed (rc={produced}): {RustCore.LastError()}");
        return produced;
    }

    private static float[] ReadF32(IntPtr p, UIntPtr n)
    {
        int count = (int)n;
        var arr = new float[count];
        if (count > 0) Marshal.Copy(p, arr, 0, count);
        return arr;
    }

    // ---- fine-tuning (LoRA) — converts this model into a trainer ----
    public void TrainerBegin(uint rank, float alpha, float dropout, string targetModulesCsv, int maxSeqLen, double learningRate)
        => Check(RustCore.rl_trainer_begin(_h, rank, alpha, dropout, targetModulesCsv, (UIntPtr)maxSeqLen, learningRate), "trainer begin");

    /// <summary>One training step on (inputIds → target); returns the loss.</summary>
    public float TrainerStep(uint[] inputIds, uint target)
    {
        Check(RustCore.rl_trainer_step(_h, inputIds, (UIntPtr)inputIds.Length, target, out float loss), "trainer step");
        return loss;
    }

    public byte[] TrainerSaveAdapter()
    {
        Check(RustCore.rl_trainer_save_adapter(_h, out IntPtr p, out UIntPtr n), "trainer save_adapter");
        try
        {
            int count = (int)n;
            var b = new byte[count];
            if (count > 0) Marshal.Copy(p, b, 0, count);
            return b;
        }
        finally { RustCore.rl_free_bytes(p, n); }
    }

    /// <summary>Request cancellation of an in-flight Generate. Thread-safe.</summary>
    public void Cancel()
    {
        if (_h != IntPtr.Zero)
            RustCore.rl_cancel(_h);
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero)
        {
            RustCore.rl_model_free(_h);
            _h = IntPtr.Zero;
        }
    }
}

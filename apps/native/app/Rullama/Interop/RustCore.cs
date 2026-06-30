using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>
/// Low-level P/Invoke bindings over <c>rust-core</c> — the C-ABI shim around the
/// published <c>rullama</c> crate (Metal/DX12/Vulkan via wgpu).
///
/// The native <c>Model</c> is <c>!Send</c>: each handle owns one OS thread for
/// its lifetime and all calls are marshalled to it inside Rust. Blocking calls
/// (load, generate) should be invoked off the UI thread. See <see cref="RustModel"/>
/// for the managed wrapper.
///
/// Library name <c>rullama_core</c> resolves to <c>librullama_core.dylib</c>
/// (macOS), <c>librullama_core.so</c> (Linux), <c>rullama_core.dll</c> (Windows).
/// </summary>
internal static class RustCore
{
    private const string Lib = "rullama_core";

    /// <summary>Per-token streaming callback: (ctx, tokenId, piece, isEos).
    /// <c>piece</c> is decoded display text valid only during the call.</summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void TokenCallback(IntPtr ctx, uint tokenId, IntPtr piece, int isEos);

    // ---- diagnostics ----
    [DllImport(Lib, EntryPoint = "rl_version")]
    internal static extern IntPtr rl_version();

    [DllImport(Lib, EntryPoint = "rl_last_error")]
    internal static extern IntPtr rl_last_error();

    [DllImport(Lib, EntryPoint = "rl_free_str")]
    internal static extern void rl_free_str(IntPtr ptr);

    [DllImport(Lib, EntryPoint = "rl_free_u32")]
    internal static extern void rl_free_u32(IntPtr ptr, UIntPtr n);

    [DllImport(Lib, EntryPoint = "rl_wgpu_probe")]
    internal static extern int rl_wgpu_probe(out IntPtr outStr);

    // ---- handle lifecycle ----
    [DllImport(Lib, EntryPoint = "rl_model_create")]
    internal static extern IntPtr rl_model_create();

    [DllImport(Lib, EntryPoint = "rl_model_free")]
    internal static extern void rl_model_free(IntPtr m);

    // ---- model + inference ----
    [DllImport(Lib, EntryPoint = "rl_model_load_path")]
    internal static extern int rl_model_load_path(
        IntPtr m, [MarshalAs(UnmanagedType.LPUTF8Str)] string path, uint maxCtx, int textOnly);

    [DllImport(Lib, EntryPoint = "rl_encode")]
    internal static extern int rl_encode(
        IntPtr m, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out IntPtr outIds, out UIntPtr outN);

    [DllImport(Lib, EntryPoint = "rl_token_str")]
    internal static extern int rl_token_str(IntPtr m, uint id, out IntPtr outStr);

    [DllImport(Lib, EntryPoint = "rl_set_sampling")]
    internal static extern int rl_set_sampling(
        IntPtr m, float temperature, uint topK, float topP, float repetitionPenalty, ulong seed);

    [DllImport(Lib, EntryPoint = "rl_reset")]
    internal static extern int rl_reset(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_vocab_size")]
    internal static extern uint rl_vocab_size(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_position")]
    internal static extern uint rl_position(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_generate")]
    internal static extern int rl_generate(
        IntPtr m, uint[] prompt, UIntPtr n, uint maxNew, TokenCallback cb, IntPtr ctx);

    [DllImport(Lib, EntryPoint = "rl_cancel")]
    internal static extern void rl_cancel(IntPtr m);

    // ---- multimodal ----
    [DllImport(Lib, EntryPoint = "rl_free_f32")]
    internal static extern void rl_free_f32(IntPtr ptr, UIntPtr n);

    [DllImport(Lib, EntryPoint = "rl_free_bytes")]
    internal static extern void rl_free_bytes(IntPtr ptr, UIntPtr n);

    // ---- fine-tuning (LoRA; reuses the model handle) ----
    [DllImport(Lib, EntryPoint = "rl_trainer_begin")]
    internal static extern int rl_trainer_begin(
        IntPtr m, uint rank, float alpha, float dropout,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string targetModules, UIntPtr maxSeqLen, double learningRate);

    [DllImport(Lib, EntryPoint = "rl_trainer_step")]
    internal static extern int rl_trainer_step(IntPtr m, uint[] inputIds, UIntPtr n, uint target, out float loss);

    [DllImport(Lib, EntryPoint = "rl_trainer_save_adapter")]
    internal static extern int rl_trainer_save_adapter(IntPtr m, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_load_adapter")]
    internal static extern int rl_load_adapter(IntPtr m, byte[] bytes, UIntPtr n);

    [DllImport(Lib, EntryPoint = "rl_clear_adapter")]
    internal static extern int rl_clear_adapter(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_rome_edit")]
    internal static extern int rl_rome_edit(
        IntPtr m, [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string subject,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string target, uint layer);

    [DllImport(Lib, EntryPoint = "rl_has_vision")]
    internal static extern int rl_has_vision(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_has_audio")]
    internal static extern int rl_has_audio(IntPtr m);

    [DllImport(Lib, EntryPoint = "rl_image_sentinel_ids")]
    internal static extern int rl_image_sentinel_ids(IntPtr m, out uint begin, out uint end);

    [DllImport(Lib, EntryPoint = "rl_audio_sentinel_ids")]
    internal static extern int rl_audio_sentinel_ids(IntPtr m, out uint begin, out uint end);

    [DllImport(Lib, EntryPoint = "rl_image_soft_token_count")]
    internal static extern long rl_image_soft_token_count(IntPtr m, UIntPtr h, UIntPtr w);

    [DllImport(Lib, EntryPoint = "rl_encode_image")]
    internal static extern int rl_encode_image(
        IntPtr m, float[] pixels, UIntPtr n, UIntPtr h, UIntPtr w, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_encode_audio")]
    internal static extern int rl_encode_audio(
        IntPtr m, float[] pcm, UIntPtr n, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_decode_wav")]
    internal static extern int rl_decode_wav(
        byte[] bytes, UIntPtr n, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_generate_spliced")]
    internal static extern int rl_generate_spliced(
        IntPtr m, uint[] prompt, UIntPtr n, uint sentinelBegin,
        float[] soft, UIntPtr softLen, UIntPtr dText, uint maxNew, TokenCallback cb, IntPtr ctx);

    // ---- text-to-speech (Kokoro; separate handle) ----
    [DllImport(Lib, EntryPoint = "rl_tts_create")]
    internal static extern IntPtr rl_tts_create();

    [DllImport(Lib, EntryPoint = "rl_tts_free")]
    internal static extern void rl_tts_free(IntPtr t);

    [DllImport(Lib, EntryPoint = "rl_tts_load_path")]
    internal static extern int rl_tts_load_path(IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, EntryPoint = "rl_tts_set_lexicon")]
    internal static extern int rl_tts_set_lexicon(
        IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string goldPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string silverPath);

    [DllImport(Lib, EntryPoint = "rl_tts_synthesize")]
    internal static extern int rl_tts_synthesize(
        IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, [MarshalAs(UnmanagedType.LPUTF8Str)] string voice,
        out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_tts_sample_rate")]
    internal static extern uint rl_tts_sample_rate(IntPtr t);

    // ---- voice cloning (StyleTTS2; separate handle) ----
    [DllImport(Lib, EntryPoint = "rl_clone_create")]
    internal static extern IntPtr rl_clone_create();

    [DllImport(Lib, EntryPoint = "rl_clone_free")]
    internal static extern void rl_clone_free(IntPtr t);

    [DllImport(Lib, EntryPoint = "rl_clone_load_path")]
    internal static extern int rl_clone_load_path(IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, EntryPoint = "rl_clone_set_lexicon")]
    internal static extern int rl_clone_set_lexicon(
        IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string goldPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string silverPath);

    [DllImport(Lib, EntryPoint = "rl_clone_encode_voice")]
    internal static extern int rl_clone_encode_voice(IntPtr t, float[] pcm, UIntPtr n, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_clone_synthesize")]
    internal static extern int rl_clone_synthesize(
        IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, float[] voice, UIntPtr voiceLen, out IntPtr outPtr, out UIntPtr outLen);

    [DllImport(Lib, EntryPoint = "rl_clone_sample_rate")]
    internal static extern uint rl_clone_sample_rate(IntPtr t);

    // ---- embeddings (EmbeddingGemma; separate handle) ----
    [DllImport(Lib, EntryPoint = "rl_embed_create")]
    internal static extern IntPtr rl_embed_create();

    [DllImport(Lib, EntryPoint = "rl_embed_free")]
    internal static extern void rl_embed_free(IntPtr t);

    [DllImport(Lib, EntryPoint = "rl_embed_load_path")]
    internal static extern int rl_embed_load_path(IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, EntryPoint = "rl_embed_dim")]
    internal static extern uint rl_embed_dim(IntPtr t);

    [DllImport(Lib, EntryPoint = "rl_embed")]
    internal static extern int rl_embed(
        IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, UIntPtr targetDim, out IntPtr outPtr, out UIntPtr outLen);

    // ---- M12: Rhai tool-orchestration (no model handle) ----
    [DllImport(Lib, EntryPoint = "rl_orch_run")]
    internal static extern int rl_orch_run(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string script,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string cachedJson,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string toolNames,
        out IntPtr outJson);

    // ---- M13: image generation (DiT diffusion; separate handle) ----
    /// <summary>Per-step progress: (ctx, step, total, stage).</summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void ImageProgressCallback(IntPtr ctx, uint step, uint total, IntPtr stage);

    [DllImport(Lib, EntryPoint = "rl_imagegen_create")]
    internal static extern IntPtr rl_imagegen_create();

    [DllImport(Lib, EntryPoint = "rl_imagegen_free")]
    internal static extern void rl_imagegen_free(IntPtr t);

    [DllImport(Lib, EntryPoint = "rl_imagegen_load_blobs")]
    internal static extern int rl_imagegen_load_blobs(IntPtr t, [MarshalAs(UnmanagedType.LPUTF8Str)] string dir);

    [DllImport(Lib, EntryPoint = "rl_imagegen_generate")]
    internal static extern int rl_imagegen_generate(
        IntPtr t,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string negPrompt,
        float cfgScale, uint latentH, uint latentW, uint steps, ulong seed,
        ImageProgressCallback progress, IntPtr ctx,
        out IntPtr outPtr, out UIntPtr outLen, out uint outW, out uint outH);

    // ---- helpers ----
    internal static string Version() => Marshal.PtrToStringUTF8(rl_version()) ?? "unknown";

    internal static string LastError()
    {
        IntPtr p = rl_last_error();
        return p == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(p) ?? string.Empty;
    }

    /// <summary>One-shot GPU probe (create + probe + free). Blocking.</summary>
    internal static string ProbeGpu()
    {
        int rc = rl_wgpu_probe(out IntPtr outStr);
        if (rc != 0)
            throw new InvalidOperationException($"rl_wgpu_probe failed (rc={rc}): {LastError()}");
        try { return Marshal.PtrToStringUTF8(outStr) ?? string.Empty; }
        finally { rl_free_str(outStr); }
    }
}

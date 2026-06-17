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

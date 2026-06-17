using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>
/// P/Invoke bindings over <c>rust-core</c> — the C-ABI shim around the
/// published <c>rullama</c> crate (Metal/DX12/Vulkan via wgpu).
///
/// The native <c>Model</c> is <c>!Send</c>: each engine handle owns one OS
/// thread for its lifetime and all calls are marshalled to it inside Rust.
/// From C# we just call the C ABI; blocking calls should be invoked off the
/// UI thread (e.g. <c>Task.Run</c>).
///
/// The library name <c>rullama_core</c> resolves per-platform to
/// <c>librullama_core.dylib</c> (macOS), <c>librullama_core.so</c> (Linux),
/// and <c>rullama_core.dll</c> (Windows).
/// </summary>
internal static class RustCore
{
    private const string Lib = "rullama_core";

    [DllImport(Lib, EntryPoint = "rl_version")]
    private static extern IntPtr rl_version();

    [DllImport(Lib, EntryPoint = "rl_last_error")]
    private static extern IntPtr rl_last_error();

    [DllImport(Lib, EntryPoint = "rl_free_str")]
    private static extern void rl_free_str(IntPtr ptr);

    [DllImport(Lib, EntryPoint = "rl_wgpu_probe")]
    private static extern int rl_wgpu_probe(out IntPtr outStr);

    /// <summary>Version string of the underlying rust-core shim.</summary>
    public static string Version() => Marshal.PtrToStringUTF8(rl_version()) ?? "unknown";

    /// <summary>Last error message for the calling thread (empty if none).</summary>
    public static string LastError()
    {
        IntPtr p = rl_last_error();
        return p == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(p) ?? string.Empty;
    }

    /// <summary>
    /// Initializes wgpu on a dedicated owning thread and returns a
    /// human-readable adapter description. Blocking — call off the UI thread.
    /// </summary>
    public static string ProbeGpu()
    {
        int rc = rl_wgpu_probe(out IntPtr outStr);
        if (rc != 0)
        {
            throw new InvalidOperationException($"rl_wgpu_probe failed (rc={rc}): {LastError()}");
        }
        try
        {
            return Marshal.PtrToStringUTF8(outStr) ?? string.Empty;
        }
        finally
        {
            rl_free_str(outStr);
        }
    }
}

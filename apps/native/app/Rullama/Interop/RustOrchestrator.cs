using System;
using System.Runtime.InteropServices;

namespace Rullama.Interop;

/// <summary>
/// Managed wrapper over the rust-core Rhai orchestrator (M12). One static call —
/// the orchestrator holds no state; the result cache lives on the C# side and is
/// passed in each pass (memoize-and-replay).
/// </summary>
public static class RustOrchestrator
{
    /// <summary>One orchestration pass. <paramref name="cachedJson"/> maps
    /// <c>"&lt;name&gt;&lt;arg&gt;"</c> to a JSON result; <paramref name="toolNames"/>
    /// is a comma-separated list of registered tool names. Returns the raw JSON
    /// envelope (<c>{"status":"needed"|"final"|"error", ...}</c>).</summary>
    public static string Run(string script, string cachedJson, string toolNames)
    {
        int rc = RustCore.rl_orch_run(script, cachedJson ?? string.Empty, toolNames ?? string.Empty, out IntPtr outJson);
        if (rc != 0)
            throw new InvalidOperationException($"rl_orch_run failed (rc={rc}): {RustCore.LastError()}");
        try { return Marshal.PtrToStringUTF8(outJson) ?? "{}"; }
        finally { RustCore.rl_free_str(outJson); }
    }
}

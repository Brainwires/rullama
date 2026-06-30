using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services.Tools;

/// <summary>
/// A tool the model can call — exposed to both the JSON <c>&lt;tool_call&gt;</c>
/// loop and the Rhai orchestrator (M12). <see cref="InvokeAsync"/> returns the
/// result as JSON text (a string, number, or object) that becomes a Rhai value.
/// </summary>
public interface ITool
{
    /// <summary>Rhai/JSON function name, e.g. <c>get_weather</c>.</summary>
    string Name { get; }

    /// <summary>One preamble line documenting the call + return shape, e.g.
    /// <c>get_weather(location)  // returns #{ summary: string }</c>.</summary>
    string PromptLine { get; }

    /// <summary>Run the tool. Returns JSON text (the orchestrator cache value).</summary>
    Task<string> InvokeAsync(string arg, CancellationToken ct);
}

/// <summary>The set of enabled tools + the orchestrator preamble/driver.</summary>
public sealed class ToolRegistry
{
    private readonly List<ITool> _tools = new();

    public IReadOnlyList<ITool> Tools => _tools;
    public ToolRegistry Add(ITool t) { _tools.Add(t); return this; }
    public ITool? Find(string name) => _tools.FirstOrDefault(t => t.Name == name);
    public string ToolNamesCsv => string.Join(",", _tools.Select(t => t.Name));

    /// <summary>The Rhai orchestrator-mode system preamble (teaches the syntax +
    /// lists the available tools). Mirrors the web orchestrator prompt.</summary>
    public string BuildPreamble()
    {
        var sb = new StringBuilder();
        sb.AppendLine("You orchestrate tools by writing a script in Rhai (a Rust-like scripting language).");
        sb.AppendLine("These tool functions are available; each returns a value:");
        sb.AppendLine();
        foreach (ITool t in _tools) sb.AppendLine("  " + t.PromptLine);
        sb.AppendLine();
        sb.AppendLine("Rhai syntax rules — Rhai is NOT Lua or Python:");
        sb.AppendLine("  - Blocks use BRACES, not then/end:   if x > 20 { ... } else { ... }");
        sb.AppendLine("  - String concatenation uses + :       \"temp is \" + temp        (NOT \"..\")");
        sb.AppendLine("  - Logical and / or are && and || :    a > 0 && b > 0             (NOT and/or)");
        sb.AppendLine("  - End statements with a semicolon ;");
        sb.AppendLine("  - The script's final expression is the returned result.");
        sb.AppendLine();
        sb.AppendLine("Reply with ONLY a Rhai script (no prose, no markdown fences) for the user's request.");
        return sb.ToString();
    }
}

/// <summary>
/// Drives the memoize-and-replay orchestration loop (M12): run the script via
/// rust-core, resolve any missing tool calls on the C# side (concurrently),
/// cache the results, and re-run until the script returns a final value.
/// </summary>
public static class ToolOrchestration
{
    private const char KeySep = '\u0001'; // matches the Rust cache key separator

    public static async Task<string> RunAsync(
        string script, ToolRegistry reg, CancellationToken ct, int maxPasses = 16)
    {
        var cache = new Dictionary<string, JsonNode?>();
        for (int pass = 0; pass < maxPasses; pass++)
        {
            ct.ThrowIfCancellationRequested();
            string cacheJson = JsonSerializer.Serialize(cache);
            string envJson = await Task.Run(() => RustOrchestrator.Run(script, cacheJson, reg.ToolNamesCsv), ct);

            using JsonDocument doc = JsonDocument.Parse(envJson);
            JsonElement root = doc.RootElement;
            string status = root.GetProperty("status").GetString() ?? "error";

            if (status == "final") return root.GetProperty("final").GetString() ?? string.Empty;
            if (status == "error")
                throw new InvalidOperationException(root.GetProperty("error").GetString() ?? "orchestrator error");

            // status == "needed": resolve each missing call concurrently.
            var needed = root.GetProperty("needed").EnumerateArray()
                .Select(e => (Name: e.GetProperty("name").GetString() ?? "",
                              Arg: e.GetProperty("arg").GetString() ?? ""))
                .ToList();
            if (needed.Count == 0)
                throw new InvalidOperationException("orchestrator returned no progress");

            IEnumerable<Task<(string Key, JsonNode? Val)>> tasks = needed.Select(async n =>
            {
                ITool? tool = reg.Find(n.Name);
                string json = tool is null
                    ? JsonSerializer.Serialize($"(unknown tool '{n.Name}')")
                    : await SafeInvoke(tool, n.Arg, ct);
                return (n.Name + KeySep + n.Arg, ParseOrString(json));
            });
            foreach ((string key, JsonNode? val) in await Task.WhenAll(tasks))
                cache[key] = val;
        }
        throw new InvalidOperationException($"orchestrator did not converge in {maxPasses} passes");
    }

    private static async Task<string> SafeInvoke(ITool tool, string arg, CancellationToken ct)
    {
        try { return await tool.InvokeAsync(arg, ct); }
        catch (Exception e) { return JsonSerializer.Serialize($"(error in {tool.Name}: {e.Message})"); }
    }

    private static JsonNode? ParseOrString(string json)
    {
        try { return JsonNode.Parse(json); }
        catch { return JsonValue.Create(json); }
    }
}

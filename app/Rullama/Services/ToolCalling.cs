using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Rullama.Services;

public sealed record ToolCall(string Name, Dictionary<string, string> Args, string Raw);

/// <summary>
/// Tolerant tool-call parsing + display cleanup (port of parseToolCalls.ts).
/// Accepts the trained JSON form {"name","arguments"}, the OpenAI {"function":…}
/// wrapper, and pythonic name(args); maps positional args via TOOL_PARAMS.
/// </summary>
public static partial class ToolCalling
{
    private static readonly Dictionary<string, string[]> ToolParams = new()
    {
        ["set_timer"] = new[] { "duration" },
        ["set_reminder"] = new[] { "text", "time" },
        ["get_weather"] = new[] { "location" },
        ["get_weather_forecast"] = new[] { "location", "days" },
        ["get_air_quality"] = new[] { "location" },
        ["get_astronomy"] = new[] { "location" },
        ["search_wikipedia"] = new[] { "query" },
        ["search_knowledge"] = new[] { "query" },
        ["get_news"] = new[] { "query" },
    };

    [GeneratedRegex(@"<tool_call\s*>?\s*([\s\S]*?)</tool_call>", RegexOptions.IgnoreCase)]
    private static partial Regex CallRegex();

    [GeneratedRegex(@"<tool_response(?:\s+for=""[^""]*"")?>[\s\S]*?</tool_response>", RegexOptions.IgnoreCase)]
    private static partial Regex ResponseRegex();

    /// <summary>All complete tool calls in the text (in order).</summary>
    public static List<ToolCall> Parse(string text)
    {
        var calls = new List<ToolCall>();
        foreach (Match m in CallRegex().Matches(text))
        {
            (string name, Dictionary<string, string> args) = ParseInner(m.Groups[1].Value.Trim());
            if (name.Length > 0)
                calls.Add(new ToolCall(name, args, m.Value));
        }
        return calls;
    }

    /// <summary>How many complete &lt;tool_response&gt; blocks the text already holds.</summary>
    public static int ResponseCount(string text) => ResponseRegex().Matches(text).Count;

    /// <summary>Strip tool_response blocks and render tool_calls as compact lines for display.</summary>
    public static string CleanForDisplay(string text)
    {
        text = ResponseRegex().Replace(text, string.Empty);
        text = CallRegex().Replace(text, m =>
        {
            (string name, Dictionary<string, string> args) = ParseInner(m.Groups[1].Value.Trim());
            if (name.Length == 0) return string.Empty;
            string a = string.Join(", ", args.Select(kv => $"{kv.Key}: {kv.Value}"));
            return $"\n> 🔧 **{name}**({a})\n";
        });
        return text.Trim();
    }

    private static (string Name, Dictionary<string, string> Args) ParseInner(string inner)
    {
        var args = new Dictionary<string, string>();
        // 1) JSON form.
        if (inner.StartsWith('{'))
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(inner);
                JsonElement root = doc.RootElement;
                if (root.TryGetProperty("function", out JsonElement fn)) root = fn;
                string name = root.TryGetProperty("name", out JsonElement n) ? n.GetString() ?? "" : "";
                if (root.TryGetProperty("arguments", out JsonElement argEl))
                {
                    if (argEl.ValueKind == JsonValueKind.Object)
                        foreach (JsonProperty p in argEl.EnumerateObject())
                            args[p.Name] = JsonScalar(p.Value);
                    else if (argEl.ValueKind == JsonValueKind.String)
                        TryParseArgsObject(argEl.GetString(), args);
                }
                return (name, args);
            }
            catch { /* fall through to pythonic */ }
        }
        // 2) Pythonic: name(arg1, key=val, …).
        Match py = PyRegex().Match(inner);
        if (py.Success)
        {
            string name = py.Groups[1].Value;
            string[] positional = ToolParams.TryGetValue(name, out string[]? p) ? p : Array.Empty<string>();
            string[] parts = SplitArgs(py.Groups[2].Value);
            int posIdx = 0;
            foreach (string raw in parts)
            {
                string part = raw.Trim();
                if (part.Length == 0) continue;
                int eq = part.IndexOf('=');
                if (eq > 0 && !part[..eq].Contains('"'))
                    args[part[..eq].Trim()] = Unquote(part[(eq + 1)..].Trim());
                else if (posIdx < positional.Length)
                    args[positional[posIdx++]] = Unquote(part);
            }
            return (name, args);
        }
        return (string.Empty, args);
    }

    private static void TryParseArgsObject(string? s, Dictionary<string, string> into)
    {
        if (string.IsNullOrWhiteSpace(s)) return;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(s);
            if (doc.RootElement.ValueKind == JsonValueKind.Object)
                foreach (JsonProperty p in doc.RootElement.EnumerateObject())
                    into[p.Name] = JsonScalar(p.Value);
        }
        catch { /* ignore */ }
    }

    private static string JsonScalar(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString() ?? "",
        _ => e.GetRawText().Trim('"'),
    };

    private static string Unquote(string s) =>
        s.Length >= 2 && ((s[0] == '"' && s[^1] == '"') || (s[0] == '\'' && s[^1] == '\'')) ? s[1..^1] : s;

    private static string[] SplitArgs(string s)
    {
        var parts = new List<string>();
        int depth = 0, start = 0; bool inStr = false; char q = '"';
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (inStr) { if (c == q) inStr = false; }
            else if (c is '"' or '\'') { inStr = true; q = c; }
            else if (c is '{' or '[' or '(') depth++;
            else if (c is '}' or ']' or ')') depth--;
            else if (c == ',' && depth == 0) { parts.Add(s[start..i]); start = i + 1; }
        }
        if (start < s.Length) parts.Add(s[start..]);
        return parts.ToArray();
    }

    [GeneratedRegex(@"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([\s\S]*)\)\s*$")]
    private static partial Regex PyRegex();
}

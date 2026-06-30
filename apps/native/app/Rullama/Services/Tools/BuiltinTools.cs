using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;

namespace Rullama.Services.Tools;

/// <summary>Weather via the existing <see cref="ToolExecutors"/> (Open-Meteo /
/// WeatherAPI). Returns an object so a Rhai script can read <c>.summary</c>.</summary>
public sealed class WeatherTool : ITool
{
    private readonly bool _fahrenheit;
    private readonly string? _apiKey;
    private readonly bool _useLocation;

    public WeatherTool(bool fahrenheit, string? apiKey, bool useLocation)
    {
        _fahrenheit = fahrenheit;
        _apiKey = apiKey;
        _useLocation = useLocation;
    }

    public string Name => "get_weather";
    public string PromptLine => "get_weather(location)      // returns #{ summary: string }";

    public async Task<string> InvokeAsync(string arg, CancellationToken ct)
    {
        var exec = new ToolExecutors { UseFahrenheit = _fahrenheit, WeatherApiKey = _apiKey, UseLocation = _useLocation };
        string summary = await exec.ExecuteAsync(new ToolCall("get_weather", new Dictionary<string, string> { ["location"] = arg }, ""));
        return new JsonObject { ["summary"] = summary }.ToJsonString();
    }
}

/// <summary>Wikipedia REST summary endpoint (keyless). Returns a string.</summary>
public sealed class WikipediaTool : ITool
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(15) };

    public string Name => "search_wikipedia";
    public string PromptLine => "search_wikipedia(query)    // returns a string";

    public async Task<string> InvokeAsync(string arg, CancellationToken ct)
    {
        string url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + Uri.EscapeDataString(arg.Replace(' ', '_'));
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Add("User-Agent", "rullama-native/0.1 (desktop)");
        using HttpResponseMessage resp = await Http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
            return JsonSerializer.Serialize($"(no Wikipedia page for '{arg}')");
        using JsonDocument doc = await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync(ct), default, ct);
        string extract = doc.RootElement.TryGetProperty("extract", out JsonElement ex) ? ex.GetString() ?? "" : "";
        return JsonSerializer.Serialize(extract.Length > 0 ? extract : $"(no summary for '{arg}')");
    }
}

/// <summary>Headlines via GNews (BYOK, like the WeatherAPI key). Returns a string.</summary>
public sealed class NewsTool : ITool
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(15) };
    private readonly string _apiKey;

    public NewsTool(string apiKey) => _apiKey = apiKey;

    public string Name => "get_news";
    public string PromptLine => "get_news(query)            // returns a string (recent headlines)";

    public async Task<string> InvokeAsync(string arg, CancellationToken ct)
    {
        string q = string.IsNullOrWhiteSpace(arg) ? "top stories" : arg;
        string url = $"https://gnews.io/api/v4/search?q={Uri.EscapeDataString(q)}&lang=en&max=5&apikey={_apiKey}";
        using HttpResponseMessage resp = await Http.GetAsync(url, ct);
        if (!resp.IsSuccessStatusCode)
            return JsonSerializer.Serialize($"(news lookup failed: HTTP {(int)resp.StatusCode})");
        using JsonDocument doc = await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync(ct), default, ct);
        if (!doc.RootElement.TryGetProperty("articles", out JsonElement arts) || arts.GetArrayLength() == 0)
            return JsonSerializer.Serialize($"(no news for '{q}')");
        var sb = new System.Text.StringBuilder();
        foreach (JsonElement a in arts.EnumerateArray())
        {
            string title = a.TryGetProperty("title", out JsonElement t) ? t.GetString() ?? "" : "";
            if (title.Length > 0) sb.Append("- ").AppendLine(title);
        }
        return JsonSerializer.Serialize(sb.ToString().TrimEnd());
    }
}

/// <summary>Retrieve grounding context from the local knowledge base (RAG).</summary>
public sealed class KnowledgeTool : ITool
{
    private readonly RagService _rag;
    public KnowledgeTool(RagService rag) => _rag = rag;

    public string Name => "search_knowledge";
    public string PromptLine => "search_knowledge(query)    // returns a string";

    public async Task<string> InvokeAsync(string arg, CancellationToken ct)
    {
        if (!RagService.Available)
            return JsonSerializer.Serialize("(knowledge base not loaded)");
        string ctx = await _rag.BuildContextAsync(arg, 5, ct);
        return JsonSerializer.Serialize(ctx.Length > 0 ? ctx : "(no relevant knowledge found)");
    }
}

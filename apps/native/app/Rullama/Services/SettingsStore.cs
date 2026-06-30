using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Rullama.Services;

/// <summary>Tiny JSON key/value settings under the app data dir (replaces the
/// PWA's localStorage). Synchronous; settings are small.</summary>
public sealed class SettingsStore
{
    private static string FilePath => Path.Combine(Paths.DataDir, "settings.json");
    private readonly Dictionary<string, string> _data;

    public SettingsStore()
    {
        try
        {
            _data = File.Exists(FilePath)
                ? JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(FilePath)) ?? new()
                : new();
        }
        catch
        {
            _data = new();
        }
    }

    public string? Get(string key) => _data.TryGetValue(key, out string? v) ? v : null;

    public void Set(string key, string value)
    {
        _data[key] = value;
        try { File.WriteAllText(FilePath, JsonSerializer.Serialize(_data)); }
        catch { /* best effort */ }
    }
}

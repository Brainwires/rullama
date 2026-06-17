using System.Collections.Generic;
using System.IO;

namespace Rullama.Services;

public sealed record CatalogModel(string Name, string Url, long Size)
{
    /// <summary>Local filename (matches the PWA's sanitized scheme).</summary>
    public string FileName => SanitizeName(Name) + ".gguf";
    public string LocalPath => Path.Combine(Paths.ModelsDir, FileName);
    public string SizeLabel => $"{Size / 1024d / 1024 / 1024:0.0} GB";
    public override string ToString() => $"{Name}  ({SizeLabel})";

    private static string SanitizeName(string name)
    {
        var chars = name.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            char c = chars[i];
            bool ok = char.IsLetterOrDigit(c) || c is '_' or '.' or '-';
            if (!ok) chars[i] = '_';
        }
        return new string(chars);
    }
}

/// <summary>Downloadable Gemma 4 models (subset of the public catalog, R2-hosted).</summary>
public static class ModelCatalog
{
    private const string Host = "models.brainwires.dev";

    public static readonly IReadOnlyList<CatalogModel> Chat = new[]
    {
        new CatalogModel("gemma4:e2b-it-qat", $"https://{Host}/gemma4-e2b-it-qat.gguf", 3349514112),
        new CatalogModel("gemma4:e2b", $"https://{Host}/gemma4-e2b.gguf", 7162394016),
        new CatalogModel("gemma4:e4b-it-qat", $"https://{Host}/gemma4-e4b-it-qat.gguf", 5154939136),
        new CatalogModel("gemma4:e4b", $"https://{Host}/gemma4-e4b.gguf", 9608338848),
    };
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services;

public readonly record struct SearchHit(string DocName, string Text, double Score);

/// <summary>
/// Retrieval-augmented generation: EmbeddingGemma + a SQLite vector store.
/// Index documents (chunk → embed → store), search by cosine, and build a
/// context preamble for the chat system turn.
/// </summary>
public sealed class RagService : IDisposable
{
    private readonly KnowledgeStore _store = new();
    private RustEmbed? _embed;
    private int _dim;
    private readonly SemaphoreSlim _loadGate = new(1, 1);

    public static string EmbedModelPath => Path.Combine(Paths.ModelsDir, "embeddinggemma-300m.gguf");
    public static bool Available => File.Exists(EmbedModelPath);

    public List<DocumentRow> Documents() => _store.ListDocuments();
    public void DeleteDocument(string id) => _store.DeleteDocument(id);

    private async Task EnsureLoadedAsync()
    {
        if (_embed is not null) return;
        await _loadGate.WaitAsync();
        try
        {
            if (_embed is not null) return;
            await Task.Run(() =>
            {
                var e = new RustEmbed();
                e.LoadPath(EmbedModelPath);
                _dim = e.Dim;
                _embed = e;
            });
        }
        finally { _loadGate.Release(); }
    }

    /// <summary>Chunk, embed, and store a document. Reports 0..1 progress.</summary>
    public async Task<int> IndexTextAsync(string name, string text, IProgress<double>? progress, CancellationToken ct)
    {
        if (!Available) throw new InvalidOperationException("Embedding model not found in the app data dir.");
        await EnsureLoadedAsync();
        List<string> chunks = TextSplit.Chunk(text);
        if (chunks.Count == 0) return 0;

        await Task.Run(() =>
        {
            string docId = _store.AddDocument(name);
            for (int i = 0; i < chunks.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                float[] v = _embed!.Embed(chunks[i], _dim);
                _store.AddChunk(docId, i, chunks[i], -1, v);
                progress?.Report((double)(i + 1) / chunks.Count);
            }
        }, ct);
        return chunks.Count;
    }

    public async Task<List<SearchHit>> SearchAsync(string query, int k, CancellationToken ct)
    {
        if (!Available) throw new InvalidOperationException("Embedding model not found.");
        await EnsureLoadedAsync();
        return await Task.Run(() =>
        {
            float[] q = _embed!.Embed(query, _dim);
            double qn = Norm(q);
            return _store.AllChunks()
                .Select(c => new SearchHit(c.DocName, c.Text, Dot(q, c.Embedding) / (qn * Norm(c.Embedding) + 1e-9)))
                .OrderByDescending(h => h.Score)
                .Take(k)
                .ToList();
        }, ct);
    }

    /// <summary>Retrieve top-k chunks and format them as a system preamble.</summary>
    public async Task<string> BuildContextAsync(string query, int k, CancellationToken ct)
    {
        List<SearchHit> hits = await SearchAsync(query, k, ct);
        if (hits.Count == 0) return string.Empty;
        var sb = new StringBuilder();
        sb.Append("Use the following context to answer. If it doesn't contain the answer, say so.\n\nContext:\n");
        int i = 1;
        foreach (SearchHit h in hits)
            sb.Append($"[{i++}] ({h.DocName}) {h.Text}\n\n");
        return sb.ToString().TrimEnd();
    }

    private static double Dot(float[] a, float[] b)
    {
        double s = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    private static double Norm(float[] a)
    {
        double s = 0;
        foreach (float x in a) s += x * x;
        return Math.Sqrt(s);
    }

    public void Dispose()
    {
        _embed?.Dispose();
        _store.Dispose();
    }
}

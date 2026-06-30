using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.Sqlite;

namespace Rullama.Services;

public readonly record struct DocumentRow(string Id, string Name, int ChunkCount);
public readonly record struct ChunkRow(string DocId, string DocName, string Text, int Page, float[] Embedding);

/// <summary>SQLite-backed knowledge store: documents + chunks with embedding BLOBs.</summary>
public sealed class KnowledgeStore : IDisposable
{
    private readonly SqliteConnection _db;

    public KnowledgeStore(string? dbPath = null)
    {
        _db = new SqliteConnection($"Data Source={dbPath ?? Path.Combine(Paths.DataDir, "knowledge.db")}");
        _db.Open();
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL, ord INTEGER NOT NULL, text TEXT NOT NULL,
                page INTEGER NOT NULL DEFAULT -1, embedding BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """;
        cmd.ExecuteNonQuery();
    }

    public string AddDocument(string name)
    {
        string id = Guid.NewGuid().ToString("N");
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = "INSERT INTO documents(id, name, created_at) VALUES($i,$n,$t)";
        cmd.Parameters.AddWithValue("$i", id);
        cmd.Parameters.AddWithValue("$n", name);
        cmd.Parameters.AddWithValue("$t", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
        cmd.ExecuteNonQuery();
        return id;
    }

    public void AddChunk(string docId, int ord, string text, int page, float[] embedding)
    {
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = "INSERT INTO chunks(doc_id, ord, text, page, embedding) VALUES($d,$o,$x,$p,$e)";
        cmd.Parameters.AddWithValue("$d", docId);
        cmd.Parameters.AddWithValue("$o", ord);
        cmd.Parameters.AddWithValue("$x", text);
        cmd.Parameters.AddWithValue("$p", page);
        cmd.Parameters.AddWithValue("$e", ToBytes(embedding));
        cmd.ExecuteNonQuery();
    }

    /// <summary>All chunks with their vectors (loaded for in-memory cosine search).</summary>
    public List<ChunkRow> AllChunks()
    {
        var rows = new List<ChunkRow>();
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = """
            SELECT c.doc_id, d.name, c.text, c.page, c.embedding
            FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id
            """;
        using SqliteDataReader r = cmd.ExecuteReader();
        while (r.Read())
        {
            using Stream s = r.GetStream(4);
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            rows.Add(new ChunkRow(r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), ToFloats(ms.ToArray())));
        }
        return rows;
    }

    public List<DocumentRow> ListDocuments()
    {
        var rows = new List<DocumentRow>();
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = """
            SELECT d.id, d.name, COUNT(c.id) FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.created_at DESC
            """;
        using SqliteDataReader r = cmd.ExecuteReader();
        while (r.Read()) rows.Add(new DocumentRow(r.GetString(0), r.GetString(1), r.GetInt32(2)));
        return rows;
    }

    public void DeleteDocument(string id)
    {
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = "DELETE FROM chunks WHERE doc_id=$i; DELETE FROM documents WHERE id=$i";
        cmd.Parameters.AddWithValue("$i", id);
        cmd.ExecuteNonQuery();
    }

    private static byte[] ToBytes(float[] v)
    {
        var b = new byte[v.Length * sizeof(float)];
        Buffer.BlockCopy(v, 0, b, 0, b.Length);
        return b;
    }

    private static float[] ToFloats(byte[] b)
    {
        var v = new float[b.Length / sizeof(float)];
        Buffer.BlockCopy(b, 0, v, 0, b.Length);
        return v;
    }

    public void Dispose() => _db.Dispose();
}

using System;
using System.Collections.Generic;
using Microsoft.Data.Sqlite;

namespace Rullama.Services;

public readonly record struct ConversationRow(string Id, string Title, long UpdatedAt);
public readonly record struct MessageRow(string Role, string Content, long CreatedAt);
public readonly record struct MessageRowId(long Id, string Role, string Content);

/// <summary>
/// SQLite-backed chat history (replaces the PWA's SQLite-in-OPFS store).
/// One conversations table + one messages table under the app data dir.
/// Calls are synchronous (local SQLite is sub-millisecond for these sizes).
/// </summary>
public sealed class ConversationStore : IDisposable
{
    private readonly SqliteConnection _db;
    // M11: the generation pump streams reply updates from the native worker
    // thread while the UI thread may read (e.g. switching conversations). The
    // single SQLite connection is not safe for concurrent use, so serialize all
    // access. lock (Monitor) is reentrant, so Rename/Touch → Update is fine.
    private readonly object _lock = new();

    public ConversationStore(string? dbPath = null)
    {
        _db = new SqliteConnection($"Data Source={dbPath ?? Paths.DbPath}");
        _db.Open();
        Exec("""
            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
            """);
    }

    private static long Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    private void Exec(string sql)
    {
        using SqliteCommand cmd = _db.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    public List<ConversationRow> List()
    {
        lock (_lock)
        {
            var rows = new List<ConversationRow>();
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC";
            using SqliteDataReader r = cmd.ExecuteReader();
            while (r.Read())
                rows.Add(new ConversationRow(r.GetString(0), r.GetString(1), r.GetInt64(2)));
            return rows;
        }
    }

    public string Create(string title)
    {
        lock (_lock)
        {
            string id = Guid.NewGuid().ToString("N");
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "INSERT INTO conversations(id, title, updated_at) VALUES($id, $t, $u)";
            cmd.Parameters.AddWithValue("$id", id);
            cmd.Parameters.AddWithValue("$t", title);
            cmd.Parameters.AddWithValue("$u", Now());
            cmd.ExecuteNonQuery();
            return id;
        }
    }

    public void Rename(string id, string title) => Update(id, title, touch: true);

    public void Touch(string id) => Update(id, null, touch: true);

    private void Update(string id, string? title, bool touch)
    {
        lock (_lock)
        {
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = title is null
                ? "UPDATE conversations SET updated_at=$u WHERE id=$id"
                : "UPDATE conversations SET title=$t, updated_at=$u WHERE id=$id";
            cmd.Parameters.AddWithValue("$id", id);
            cmd.Parameters.AddWithValue("$u", Now());
            if (title is not null) cmd.Parameters.AddWithValue("$t", title);
            cmd.ExecuteNonQuery();
        }
    }

    public void Delete(string id)
    {
        lock (_lock)
        {
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "DELETE FROM messages WHERE conv_id=$id; DELETE FROM conversations WHERE id=$id";
            cmd.Parameters.AddWithValue("$id", id);
            cmd.ExecuteNonQuery();
        }
    }

    public List<MessageRow> Messages(string convId)
    {
        lock (_lock)
        {
            var rows = new List<MessageRow>();
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "SELECT role, content, created_at FROM messages WHERE conv_id=$id ORDER BY id";
            cmd.Parameters.AddWithValue("$id", convId);
            using SqliteDataReader r = cmd.ExecuteReader();
            while (r.Read())
                rows.Add(new MessageRow(r.GetString(0), r.GetString(1), r.GetInt64(2)));
            return rows;
        }
    }

    /// <summary>All messages for a conversation with their row ids (ordered).</summary>
    public List<MessageRowId> MessagesWithIds(string convId)
    {
        lock (_lock)
        {
            var rows = new List<MessageRowId>();
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "SELECT id, role, content FROM messages WHERE conv_id=$id ORDER BY id";
            cmd.Parameters.AddWithValue("$id", convId);
            using SqliteDataReader r = cmd.ExecuteReader();
            while (r.Read())
                rows.Add(new MessageRowId(r.GetInt64(0), r.GetString(1), r.GetString(2)));
            return rows;
        }
    }

    public long AddMessage(string convId, string role, string content)
    {
        lock (_lock)
        {
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText =
                "INSERT INTO messages(conv_id, role, content, created_at) VALUES($c, $r, $t, $ts); SELECT last_insert_rowid();";
            cmd.Parameters.AddWithValue("$c", convId);
            cmd.Parameters.AddWithValue("$r", role);
            cmd.Parameters.AddWithValue("$t", content);
            cmd.Parameters.AddWithValue("$ts", Now());
            return (long)(cmd.ExecuteScalar() ?? 0L);
        }
    }

    public void UpdateMessage(long messageId, string content)
    {
        lock (_lock)
        {
            using SqliteCommand cmd = _db.CreateCommand();
            cmd.CommandText = "UPDATE messages SET content=$t WHERE id=$id";
            cmd.Parameters.AddWithValue("$id", messageId);
            cmd.Parameters.AddWithValue("$t", content);
            cmd.ExecuteNonQuery();
        }
    }

    public void Dispose() => _db.Dispose();
}

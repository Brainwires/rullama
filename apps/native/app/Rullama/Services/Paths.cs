using System;
using System.IO;

namespace Rullama.Services;

/// <summary>Per-platform application data locations.</summary>
public static class Paths
{
    /// <summary>App data dir (created if missing): %LOCALAPPDATA%\Rullama (Win),
    /// ~/.local/share/Rullama (Linux), ~/Library/Application Support/Rullama is
    /// not used — LocalApplicationData keeps it consistent cross-platform.</summary>
    public static string DataDir
    {
        get
        {
            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Rullama");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string DbPath => Path.Combine(DataDir, "rullama.db");

    public static string ModelsDir
    {
        get
        {
            string dir = Path.Combine(DataDir, "models");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    /// <summary>Background generation queue (M11): manifest + per-job media sidecars.</summary>
    public static string QueueDir
    {
        get
        {
            string dir = Path.Combine(DataDir, "queue");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }
}

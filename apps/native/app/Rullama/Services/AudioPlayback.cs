using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Rullama.Services;

/// <summary>
/// Minimal cross-platform PCM playback: writes a temp WAV and plays it via the
/// OS (afplay / aplay / PowerShell SoundPlayer). Interim — a proper audio
/// library (capture + low-latency playback) arrives with mic/STT in M4c.
/// </summary>
public sealed class AudioPlayback
{
    private Process? _proc;

    public async Task PlayAsync(float[] pcm, int sampleRate, CancellationToken ct)
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"rullama_tts_{Guid.NewGuid():N}.wav");
        WriteWav(tmp, pcm, sampleRate);
        try { await PlayFileAsync(tmp, ct); }
        finally { try { File.Delete(tmp); } catch { /* best effort */ } }
    }

    public void Stop()
    {
        try { _proc?.Kill(entireProcessTree: true); } catch { /* */ }
        _proc = null;
    }

    private async Task PlayFileAsync(string path, CancellationToken ct)
    {
        (string exe, string args) = GetPlayer(path);
        var psi = new ProcessStartInfo(exe, args) { UseShellExecute = false, CreateNoWindow = true };
        var proc = new Process { StartInfo = psi };
        _proc = proc;
        proc.Start();
        try { await proc.WaitForExitAsync(ct); }
        catch (OperationCanceledException) { try { proc.Kill(true); } catch { } throw; }
        finally { _proc = null; }
    }

    private static (string Exe, string Args) GetPlayer(string path)
    {
        if (OperatingSystem.IsMacOS()) return ("afplay", $"\"{path}\"");
        if (OperatingSystem.IsWindows())
            return ("powershell", $"-NoProfile -Command \"(New-Object Media.SoundPlayer '{path}').PlaySync()\"");
        return ("aplay", $"\"{path}\""); // Linux (PulseAudio users may prefer paplay)
    }

    private static void WriteWav(string path, float[] pcm, int sampleRate)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var w = new BinaryWriter(fs);
        int dataBytes = pcm.Length * 2;
        w.Write("RIFF"u8.ToArray());
        w.Write(36 + dataBytes);
        w.Write("WAVE"u8.ToArray());
        w.Write("fmt "u8.ToArray());
        w.Write(16);
        w.Write((short)1);
        w.Write((short)1);
        w.Write(sampleRate);
        w.Write(sampleRate * 2);
        w.Write((short)2);
        w.Write((short)16);
        w.Write("data"u8.ToArray());
        w.Write(dataBytes);
        foreach (float s in pcm)
            w.Write((short)(Math.Clamp(s, -1f, 1f) * short.MaxValue));
    }
}

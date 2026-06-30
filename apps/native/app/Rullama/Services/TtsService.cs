using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services;

/// <summary>
/// Shared Kokoro TTS: lazily loads the model + lexicon from the app data dir,
/// synthesizes on its native thread, and plays through <see cref="AudioPlayback"/>.
/// </summary>
public sealed class TtsService : IDisposable
{
    private readonly AudioPlayback _audio = new();
    private RustTts? _tts;
    private readonly SemaphoreSlim _loadGate = new(1, 1);

    public static string KokoroPath => Path.Combine(Paths.ModelsDir, "kokoro-82m.gguf");
    public static string GoldPath => Path.Combine(Paths.ModelsDir, "us_gold.json");
    public static string SilverPath => Path.Combine(Paths.ModelsDir, "us_silver.json");

    /// <summary>True if the TTS assets are present (model + lexicon).</summary>
    public static bool Available => File.Exists(KokoroPath) && File.Exists(GoldPath) && File.Exists(SilverPath);

    private async Task EnsureLoadedAsync()
    {
        if (_tts is not null) return;
        await _loadGate.WaitAsync();
        try
        {
            if (_tts is not null) return;
            await Task.Run(() =>
            {
                var t = new RustTts();
                t.LoadPath(KokoroPath);
                t.SetLexicon(GoldPath, SilverPath);
                _tts = t;
            });
        }
        finally { _loadGate.Release(); }
    }

    /// <summary>Synthesize and play. Blocking work runs off the UI thread.</summary>
    public async Task SpeakAsync(string text, string voice, CancellationToken ct)
    {
        if (!Available) throw new InvalidOperationException("TTS assets not found in the app data dir.");
        await EnsureLoadedAsync();
        ct.ThrowIfCancellationRequested();
        float[] pcm = await Task.Run(() => _tts!.Synthesize(text, voice), ct);
        await _audio.PlayAsync(pcm, _tts!.SampleRate, ct);
    }

    public void Stop() => _audio.Stop();

    public void Dispose() => _tts?.Dispose();
}

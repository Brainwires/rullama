using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services;

/// <summary>
/// Zero-shot voice cloning (StyleTTS2): encode a reference clip into a voice
/// vector and synthesize new text in that voice, then play it.
/// </summary>
public sealed class CloneService : IDisposable
{
    private readonly AudioPlayback _audio = new();
    private RustClone? _clone;
    private float[]? _voice;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public static string ModelPath => Path.Combine(Paths.ModelsDir, "styletts2-libritts-f16.gguf");
    public static string GoldPath => Path.Combine(Paths.ModelsDir, "us_gold.json");
    public static string SilverPath => Path.Combine(Paths.ModelsDir, "us_silver.json");
    public static bool Available => File.Exists(ModelPath) && File.Exists(GoldPath) && File.Exists(SilverPath);

    private async Task EnsureLoadedAsync()
    {
        if (_clone is not null) return;
        await _gate.WaitAsync();
        try
        {
            if (_clone is not null) return;
            await Task.Run(() =>
            {
                var c = new RustClone();
                c.LoadPath(ModelPath);
                c.SetLexicon(GoldPath, SilverPath);
                _clone = c;
            });
        }
        finally { _gate.Release(); }
    }

    /// <summary>Encode a reference WAV into the active cloned voice.</summary>
    public async Task CreateVoiceAsync(byte[] referenceWav, CancellationToken ct)
    {
        if (!Available) throw new InvalidOperationException("StyleTTS2 model not found in the app data dir.");
        await EnsureLoadedAsync();
        float[] pcm = RustModel.DecodeWav(referenceWav);
        _voice = await Task.Run(() => _clone!.EncodeVoice(pcm), ct);
    }

    /// <summary>Synthesize + play text in the cloned voice (requires CreateVoiceAsync first).</summary>
    public async Task SpeakAsync(string text, CancellationToken ct)
    {
        if (_voice is null) throw new InvalidOperationException("Create a voice from a reference clip first.");
        float[] pcm = await Task.Run(() => _clone!.Synthesize(text, _voice!), ct);
        await _audio.PlayAsync(pcm, _clone!.SampleRate, ct);
    }

    public bool HasVoice => _voice is not null;
    public void Stop() => _audio.Stop();
    public void Dispose() => _clone?.Dispose();
}

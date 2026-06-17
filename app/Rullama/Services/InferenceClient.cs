using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services;

/// <summary>
/// Owns the loaded <see cref="RustModel"/> and drives chat turns. Mirrors the
/// PWA's inference contract: render → encode → streaming generate. Blocking
/// engine calls run on the thread pool; the per-token callback fires on the
/// native worker thread, so callers marshal pieces to the UI themselves.
/// </summary>
public sealed class InferenceClient : IDisposable
{
    private RustModel? _model;
    private readonly object _gate = new();

    public bool IsLoaded => _model is not null;
    public uint VocabSize => _model?.VocabSize() ?? 0;

    /// <summary>Load a GGUF by path (off the UI thread). Replaces any prior model.
    /// Loads vision/audio towers by default so multimodal works; pass
    /// textOnly:true on memory-constrained targets.</summary>
    public async Task LoadAsync(string path, uint maxContext = 2048, bool textOnly = false)
    {
        await Task.Run(() =>
        {
            var m = new RustModel();
            try
            {
                m.LoadPath(path, maxContext, textOnly);
                // PWA defaults.
                m.SetSampling(temperature: 1.0f, topK: 64, topP: 0.95f, repetitionPenalty: 1.3f, seed: 0);
            }
            catch
            {
                m.Dispose();
                throw;
            }
            lock (_gate)
            {
                _model?.Dispose();
                _model = m;
            }
        });
    }

    public void SetSampling(float temperature, uint topK, float topP, float repetitionPenalty, ulong seed)
    {
        lock (_gate) _model?.SetSampling(temperature, topK, topP, repetitionPenalty, seed);
    }

    /// <summary>
    /// Generate a reply for the given history (role/content pairs), streaming
    /// decoded pieces to <paramref name="onPiece"/>. Cancellable via
    /// <paramref name="ct"/>. Returns the number of tokens produced.
    /// </summary>
    public async Task<int> SendAsync(
        IReadOnlyList<(string Role, string Content)> history,
        uint maxNew,
        Action<string> onPiece,
        CancellationToken ct)
    {
        RustModel model = _model ?? throw new InvalidOperationException("No model loaded.");
        return await Task.Run(() =>
        {
            // M1: full re-prefill each turn (no KV reuse yet).
            model.Reset();
            string prompt = BuildPrompt(history);
            uint[] ids = model.Encode(prompt);
            using CancellationTokenRegistration reg = ct.Register(model.Cancel);
            return model.Generate(ids, maxNew, onPiece);
        }, ct);
    }

    public bool HasVision => _model?.HasVision() ?? false;

    /// <summary>
    /// Generate a reply where the latest user turn includes one image. The
    /// history's user content must contain the &lt;|image&gt;&lt;image|&gt; marker pair;
    /// the image's soft tokens are spliced at the begin sentinel during prefill.
    /// </summary>
    public async Task<int> SendImageAsync(
        IReadOnlyList<(string Role, string Content)> history,
        float[] pixels, int h, int w,
        uint maxNew,
        Action<string> onPiece,
        CancellationToken ct)
    {
        RustModel model = _model ?? throw new InvalidOperationException("No model loaded.");
        return await Task.Run(() =>
        {
            (uint Begin, uint End)? sent = model.ImageSentinels()
                ?? throw new InvalidOperationException("This model has no vision tower.");
            float[] soft = model.EncodeImage(pixels, h, w);
            long nSoft = model.ImageSoftTokenCount(h, w);
            int dText = nSoft > 0 ? soft.Length / (int)nSoft : soft.Length;

            model.Reset();
            string prompt = BuildPrompt(history);
            uint[] ids = model.Encode(prompt);
            using CancellationTokenRegistration reg = ct.Register(model.Cancel);
            return model.GenerateSpliced(ids, sent.Value.Begin, soft, dText, maxNew, onPiece);
        }, ct);
    }

    /// <summary>
    /// Render history to the exact Gemma 4 chat format the crate tokenizes
    /// (mirrors template::gemma4_small::render_for_completion): a leading BOS,
    /// each turn wrapped in &lt;|turn&gt;role\n…&lt;turn|&gt;\n, then an open model turn.
    /// </summary>
    private static string BuildPrompt(IReadOnlyList<(string Role, string Content)> history)
    {
        var sb = new StringBuilder();
        sb.Append("<bos>");
        foreach ((string role, string content) in history)
        {
            sb.Append("<|turn>").Append(role).Append('\n').Append(content).Append("<turn|>\n");
        }
        sb.Append("<|turn>model\n");
        return sb.ToString();
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _model?.Dispose();
            _model = null;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Rullama.Interop;

namespace Rullama.Services;

/// <summary>
/// LoRA fine-tuning on its own model handle (separate from chat, so chat stays
/// usable). Loads a base model text-only, converts it to a trainer, and steps
/// over (prompt → first-completion-token) examples (NextToken objective).
/// </summary>
public sealed class TrainingService : IDisposable
{
    private RustModel? _model; // holds the trainer after RunAsync

    public async Task RunAsync(
        string modelPath,
        IReadOnlyList<(string Prompt, string Completion)> examples,
        uint rank, double learningRate, int maxSeqLen, int steps,
        Action<int, float> onStep,
        CancellationToken ct)
    {
        await Task.Run(() =>
        {
            var m = new RustModel();
            try
            {
                m.LoadPath(modelPath, maxContext: (uint)Math.Max(16, maxSeqLen + 8), textOnly: true);
                var pairs = examples
                    .Select(e => (input: m.Encode(e.Prompt), target: FirstToken(m, e.Completion)))
                    .Where(p => p.input.Length > 0)
                    .ToList();
                if (pairs.Count == 0) throw new InvalidOperationException("No usable examples.");

                m.TrainerBegin(rank, rank * 2f, 0.0f, "attn_q,attn_v", maxSeqLen, learningRate);
                for (int s = 0; s < steps && !ct.IsCancellationRequested; s++)
                {
                    (uint[] input, uint target) = pairs[s % pairs.Count];
                    float loss = m.TrainerStep(input, target);
                    onStep(s, loss);
                }
                _model?.Dispose();
                _model = m;
            }
            catch
            {
                m.Dispose();
                throw;
            }
        }, ct);
    }

    public async Task<string> SaveAdapterAsync(string path)
    {
        RustModel m = _model ?? throw new InvalidOperationException("Nothing trained yet.");
        return await Task.Run(() =>
        {
            byte[] bytes = m.TrainerSaveAdapter();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, bytes);
            return path;
        });
    }

    private static uint FirstToken(RustModel m, string completion)
    {
        uint[] ids = m.Encode(completion.StartsWith(' ') ? completion : " " + completion);
        return ids.Length > 0 ? ids[0] : 0;
    }

    public void Dispose() => _model?.Dispose();
}

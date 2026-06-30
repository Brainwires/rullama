using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Rullama.Services;

public enum JobStatus { Queued, Running }

/// <summary>
/// One enqueued generation (M11). Generation tunables and attachments are
/// captured at enqueue time, so a queued job runs with the settings that were in
/// effect when the user pressed Send. Mirrors the PWA's <c>GenJob</c>.
/// </summary>
public sealed class GenJob
{
    public string JobId { get; init; } = Guid.NewGuid().ToString("N");
    public string ConvId { get; init; } = string.Empty;

    /// <summary>SQLite row id of the (initially empty) model reply row.</summary>
    public long ModelRowId { get; set; }

    public string UserText { get; init; } = string.Empty;
    public long CreatedAt { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    // ---- captured generation tunables ----
    public float Temperature { get; init; } = 1.0f;
    public uint TopK { get; init; } = 64;
    public float TopP { get; init; } = 0.95f;
    public float RepetitionPenalty { get; init; } = 1.3f;
    public uint MaxTokens { get; init; } = 1024;
    public bool Thinking { get; init; }
    public bool ToolMode { get; init; }
    /// <summary>M12: run the model-authored Rhai orchestration loop (with the JSON
    /// tool loop as a fallback) instead of the plain JSON tool loop.</summary>
    public bool OrchestratorMode { get; init; }

    /// <summary>System content built at enqueue time (already includes the tool
    /// schema, thinking prefix, and any RAG preamble).</summary>
    public string SystemContent { get; init; } = string.Empty;

    public bool UseFahrenheit { get; init; }
    public bool UseLocation { get; init; }
    public string? WeatherApiKey { get; init; }

    // ---- attachments (raw, in-memory while live; persisted as .f32 sidecars) ----
    public float[]? ImagePixels { get; set; }
    public int ImageH { get; init; }
    public int ImageW { get; init; }
    public float[]? AudioPcm { get; set; }

    public bool HasImage => ImagePixels is not null || ImageH > 0;
    public bool HasAudio => AudioPcm is not null;

    public JobStatus Status { get; set; } = JobStatus.Queued;
}

/// <summary>
/// Serial FIFO generation pump (M11). The native model is <c>!Send</c> with one
/// owning thread, so generation is inherently serial; this keeps one job running
/// in the background while the UI is free to browse/queue. The pump runs on the
/// UI thread (Avalonia sync context); <c>_runner</c>'s awaits resume there, so
/// the job list is only ever touched on the UI thread.
/// </summary>
public sealed class GenerationQueue
{
    private readonly List<GenJob> _jobs = new();
    private readonly Func<GenJob, CancellationToken, Task> _runner;
    private bool _pumping;
    private CancellationTokenSource? _runningCts;

    public GenerationQueue(Func<GenJob, CancellationToken, Task> runner) => _runner = runner;

    /// <summary>Raised after any change to the queue (enqueue / status / removal).</summary>
    public event Action? Changed;

    public bool Active { get; private set; }
    public string? RunningConvId { get; private set; }
    public IReadOnlyList<GenJob> Jobs => _jobs;

    public bool HasPendingFor(string convId) => _jobs.Any(j => j.ConvId == convId);
    public bool IsQueuedConv(string convId) =>
        _jobs.Any(j => j.Status == JobStatus.Queued && j.ConvId == convId);

    public void Enqueue(GenJob job, bool kick = true)
    {
        _jobs.Add(job);
        Changed?.Invoke();
        if (kick) Kick();
    }

    public bool Remove(GenJob job)
    {
        bool removed = _jobs.Remove(job);
        if (removed) Changed?.Invoke();
        return removed;
    }

    /// <summary>Cancel the currently-running job (if any).</summary>
    public void CancelRunning() => _runningCts?.Cancel();

    public void Kick()
    {
        if (_pumping) return;
        _pumping = true;
        Active = true;
        Changed?.Invoke();
        _ = PumpAsync();
    }

    private async Task PumpAsync()
    {
        try
        {
            while (true)
            {
                GenJob? job = _jobs.FirstOrDefault(j => j.Status == JobStatus.Queued);
                if (job is null) break;

                job.Status = JobStatus.Running;
                RunningConvId = job.ConvId;
                _runningCts = new CancellationTokenSource();
                Changed?.Invoke();
                try
                {
                    await _runner(job, _runningCts.Token);
                }
                catch
                {
                    // per-job failure is surfaced by the runner; keep the pump alive.
                }
                finally
                {
                    _runningCts?.Dispose();
                    _runningCts = null;
                    RunningConvId = null;
                    _jobs.Remove(job);
                    Changed?.Invoke();
                }
            }
        }
        finally
        {
            _pumping = false;
            Active = false;
            Changed?.Invoke();
        }
    }
}

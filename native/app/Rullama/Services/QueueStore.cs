using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Rullama.Services;

/// <summary>
/// Persists the generation queue (M11) so queued work — including attachment
/// pixels/PCM — survives an app restart. Replaces the PWA's OPFS
/// <c>rullama-queue/</c> layout with files under the app data dir:
/// <code>
/// queue/manifest.json        ordered jobs (metadata only, FIFO by CreatedAt)
/// queue/&lt;jobId&gt;-img.f32      channel-first f32 image pixels
/// queue/&lt;jobId&gt;-aud.f32      mono f32 PCM
/// </code>
/// API keys / secrets are never written here (only the metadata below).
/// </summary>
public sealed class QueueStore
{
    private static string Dir => Paths.QueueDir;
    private static string ManifestPath => Path.Combine(Dir, "manifest.json");
    private static string ImgPath(string jobId) => Path.Combine(Dir, jobId + "-img.f32");
    private static string AudPath(string jobId) => Path.Combine(Dir, jobId + "-aud.f32");

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = false };

    private sealed record JobDto(
        string JobId, string ConvId, long ModelRowId, string UserText, long CreatedAt,
        float Temperature, uint TopK, float TopP, float RepetitionPenalty, uint MaxTokens,
        bool Thinking, bool ToolMode, string SystemContent,
        bool UseFahrenheit, bool UseLocation, string? WeatherApiKey,
        bool HasImage, int ImageH, int ImageW, bool HasAudio);

    /// <summary>Rewrite the manifest from the live job list (metadata only).</summary>
    public void Persist(IEnumerable<GenJob> jobs)
    {
        var metas = jobs.Select(j => new JobDto(
            j.JobId, j.ConvId, j.ModelRowId, j.UserText, j.CreatedAt,
            j.Temperature, j.TopK, j.TopP, j.RepetitionPenalty, j.MaxTokens,
            j.Thinking, j.ToolMode, j.SystemContent,
            j.UseFahrenheit, j.UseLocation, j.WeatherApiKey,
            j.HasImage, j.ImageH, j.ImageW, j.HasAudio)).ToList();
        File.WriteAllText(ManifestPath, JsonSerializer.Serialize(metas, JsonOpts));
    }

    /// <summary>Write a job's attachment media to per-job .f32 sidecars.</summary>
    public void SaveJobMedia(GenJob job)
    {
        if (job.ImagePixels is { } px) WriteF32(ImgPath(job.JobId), px);
        if (job.AudioPcm is { } pcm) WriteF32(AudPath(job.JobId), pcm);
    }

    /// <summary>Load the persisted queue, reattaching media per job (FIFO order).</summary>
    public List<GenJob> Load()
    {
        var jobs = new List<GenJob>();
        if (!File.Exists(ManifestPath)) return jobs;
        List<JobDto>? metas;
        try { metas = JsonSerializer.Deserialize<List<JobDto>>(File.ReadAllText(ManifestPath), JsonOpts); }
        catch { return jobs; }
        if (metas is null) return jobs;

        foreach (JobDto m in metas.OrderBy(m => m.CreatedAt))
        {
            var job = new GenJob
            {
                JobId = m.JobId,
                ConvId = m.ConvId,
                ModelRowId = m.ModelRowId,
                UserText = m.UserText,
                CreatedAt = m.CreatedAt,
                Temperature = m.Temperature,
                TopK = m.TopK,
                TopP = m.TopP,
                RepetitionPenalty = m.RepetitionPenalty,
                MaxTokens = m.MaxTokens,
                Thinking = m.Thinking,
                ToolMode = m.ToolMode,
                SystemContent = m.SystemContent,
                UseFahrenheit = m.UseFahrenheit,
                UseLocation = m.UseLocation,
                WeatherApiKey = m.WeatherApiKey,
                ImageH = m.ImageH,
                ImageW = m.ImageW,
            };
            if (m.HasImage && File.Exists(ImgPath(m.JobId))) job.ImagePixels = ReadF32(ImgPath(m.JobId));
            if (m.HasAudio && File.Exists(AudPath(m.JobId))) job.AudioPcm = ReadF32(AudPath(m.JobId));
            jobs.Add(job);
        }
        return jobs;
    }

    public void DropJobMedia(string jobId)
    {
        TryDelete(ImgPath(jobId));
        TryDelete(AudPath(jobId));
    }

    public void Clear()
    {
        try { if (Directory.Exists(Dir)) Directory.Delete(Dir, recursive: true); }
        catch { /* best effort */ }
    }

    private static void WriteF32(string path, float[] data)
    {
        byte[] bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] data = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, data, 0, data.Length * sizeof(float));
        return data;
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort */ }
    }
}

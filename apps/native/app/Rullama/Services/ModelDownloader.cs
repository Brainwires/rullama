using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;

namespace Rullama.Services;

/// <summary>Resumable HTTP-range model downloader → app models dir (mirrors the
/// PWA's Range loader). Writes to a .part file and resumes from it.</summary>
public sealed class ModelDownloader
{
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    /// <summary>Download <paramref name="m"/> if not already present; returns the local path.</summary>
    public async Task<string> DownloadAsync(CatalogModel m, IProgress<double> progress, CancellationToken ct)
    {
        string dest = m.LocalPath;
        if (File.Exists(dest) && new FileInfo(dest).Length == m.Size)
        {
            progress.Report(1.0);
            return dest;
        }

        string part = dest + ".part";
        long have = File.Exists(part) ? new FileInfo(part).Length : 0;

        using var req = new HttpRequestMessage(HttpMethod.Get, m.Url);
        if (have > 0) req.Headers.Range = new RangeHeaderValue(have, null);

        using HttpResponseMessage resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();

        bool resume = resp.StatusCode == HttpStatusCode.PartialContent && have > 0;
        long total = m.Size > 0 ? m.Size : (resp.Content.Headers.ContentLength ?? 0) + (resume ? have : 0);
        long written = resume ? have : 0;

        await using (var fs = new FileStream(part, resume ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None))
        await using (Stream src = await resp.Content.ReadAsStreamAsync(ct))
        {
            var buf = new byte[1 << 20];
            int read;
            while ((read = await src.ReadAsync(buf, ct)) > 0)
            {
                await fs.WriteAsync(buf.AsMemory(0, read), ct);
                written += read;
                if (total > 0) progress.Report((double)written / total);
            }
        }

        File.Move(part, dest, overwrite: true);
        progress.Report(1.0);
        return dest;
    }
}

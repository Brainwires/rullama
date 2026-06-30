using Rullama.Interop;

// Verifies Kokoro TTS: load model + lexicon, synthesize text → PCM, write a WAV.
// Defaults to the app data dir; override via env.

string dataDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Rullama", "models");
string model = Environment.GetEnvironmentVariable("RULLAMA_TEST_KOKORO") ?? Path.Combine(dataDir, "kokoro-82m.gguf");
string gold = Environment.GetEnvironmentVariable("RULLAMA_TEST_GOLD") ?? Path.Combine(dataDir, "us_gold.json");
string silver = Environment.GetEnvironmentVariable("RULLAMA_TEST_SILVER") ?? Path.Combine(dataDir, "us_silver.json");

foreach ((string label, string p) in new[] { ("model", model), ("gold", gold), ("silver", silver) })
{
    if (!File.Exists(p)) { Console.Error.WriteLine($"missing {label}: {p}"); return 1; }
}

using var tts = new RustTts();
Console.WriteLine("loading Kokoro…");
tts.LoadPath(model);
tts.SetLexicon(gold, silver);
Console.WriteLine($"sample rate = {tts.SampleRate} Hz; synthesizing…");

float[] pcm = tts.Synthesize("Hello, this is rullama speaking entirely on device.", "af_heart");
double seconds = (double)pcm.Length / tts.SampleRate;
Console.WriteLine($"PCM samples = {pcm.Length} ({seconds:0.00}s)");
if (pcm.Length == 0) { Console.Error.WriteLine("no audio produced"); return 1; }

string outPath = "/tmp/rullama_tts.wav";
WriteWav(outPath, pcm, tts.SampleRate);
Console.WriteLine($"wrote {outPath} ({new FileInfo(outPath).Length} bytes)");
return 0;

static void WriteWav(string path, float[] pcm, int sampleRate)
{
    using var fs = new FileStream(path, FileMode.Create);
    using var w = new BinaryWriter(fs);
    int dataBytes = pcm.Length * 2; // 16-bit mono
    w.Write("RIFF"u8.ToArray());
    w.Write(36 + dataBytes);
    w.Write("WAVE"u8.ToArray());
    w.Write("fmt "u8.ToArray());
    w.Write(16);                 // PCM chunk size
    w.Write((short)1);           // PCM
    w.Write((short)1);           // mono
    w.Write(sampleRate);
    w.Write(sampleRate * 2);     // byte rate
    w.Write((short)2);           // block align
    w.Write((short)16);          // bits per sample
    w.Write("data"u8.ToArray());
    w.Write(dataBytes);
    foreach (float s in pcm)
    {
        int v = (int)(Math.Clamp(s, -1f, 1f) * short.MaxValue);
        w.Write((short)v);
    }
}

using Rullama.Interop;

// Verifies StyleTTS2 voice cloning: encode a reference clip → voice vector,
// synthesize new text in that voice, write a WAV. Reference defaults to the
// Kokoro TtsSmoke output (/tmp/rullama_tts.wav).

string dataDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Rullama", "models");
string model = Environment.GetEnvironmentVariable("RULLAMA_TEST_CLONE") ?? Path.Combine(dataDir, "styletts2-libritts-f16.gguf");
string gold = Path.Combine(dataDir, "us_gold.json");
string silver = Path.Combine(dataDir, "us_silver.json");
string refWav = Environment.GetEnvironmentVariable("RULLAMA_TEST_REFWAV") ?? "/tmp/rullama_tts.wav";

foreach ((string label, string p) in new[] { ("model", model), ("gold", gold), ("silver", silver), ("ref wav", refWav) })
    if (!File.Exists(p)) { Console.Error.WriteLine($"missing {label}: {p}"); return 1; }

using var clone = new RustClone();
Console.WriteLine("loading StyleTTS2…");
clone.LoadPath(model);
clone.SetLexicon(gold, silver);

float[] refPcm = RustModel.DecodeWav(File.ReadAllBytes(refWav));
Console.WriteLine($"reference {refPcm.Length} samples ({refPcm.Length / 24000.0:0.0}s); encoding voice…");
float[] voice = clone.EncodeVoice(refPcm);
Console.WriteLine($"voice vector = {voice.Length}; synthesizing…");

float[] pcm = clone.Synthesize("This is my cloned voice speaking entirely on device.", voice);
Console.WriteLine($"synth = {pcm.Length} samples ({pcm.Length / (double)clone.SampleRate:0.0}s)");
if (pcm.Length == 0) { Console.Error.WriteLine("no audio"); return 1; }

WriteWav("/tmp/rullama_clone.wav", pcm, clone.SampleRate);
Console.WriteLine("wrote /tmp/rullama_clone.wav");
return 0;

static void WriteWav(string path, float[] pcm, int sampleRate)
{
    using var fs = new FileStream(path, FileMode.Create);
    using var w = new BinaryWriter(fs);
    int dataBytes = pcm.Length * 2;
    w.Write("RIFF"u8.ToArray()); w.Write(36 + dataBytes); w.Write("WAVE"u8.ToArray());
    w.Write("fmt "u8.ToArray()); w.Write(16); w.Write((short)1); w.Write((short)1);
    w.Write(sampleRate); w.Write(sampleRate * 2); w.Write((short)2); w.Write((short)16);
    w.Write("data"u8.ToArray()); w.Write(dataBytes);
    foreach (float s in pcm) w.Write((short)(Math.Clamp(s, -1f, 1f) * short.MaxValue));
}

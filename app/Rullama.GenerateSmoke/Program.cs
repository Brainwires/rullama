using Rullama.Services;

// Fast smoke for the production InferenceClient path.
//   RULLAMA_TEST_GGUF  – required, path to a Gemma 4 GGUF (e.g. an Ollama e2b blob)
//   RULLAMA_TEST_VISION=1 – also load the vision tower and run an image test
//                           (slower: keeps the towers resident).

string? path = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF");
if (path is null)
{
    Console.Error.WriteLine("Set RULLAMA_TEST_GGUF to a Gemma 4 GGUF path.");
    return 1;
}
bool visionMode = Environment.GetEnvironmentVariable("RULLAMA_TEST_VISION") == "1";
bool audioOnly = Environment.GetEnvironmentVariable("RULLAMA_TEST_AUDIO_ONLY") == "1";
bool full = visionMode || audioOnly;

using var engine = new InferenceClient();
Console.WriteLine($"loading ({(full ? "full towers" : "text-only")}, ctx 1024)…");
await engine.LoadAsync(path, maxContext: 1024, textOnly: !full);
Console.WriteLine($"ready · vocab {engine.VocabSize}");

if (!audioOnly)
{
    Console.WriteLine("--- reply ---");
    var history = new List<(string, string)> { ("user", "Name three primary colors.") };
    int n = await engine.SendAsync(history, maxNew: 48,
        onPiece: piece => { Console.Write(piece); Console.Out.Flush(); }, ct: CancellationToken.None);
    Console.WriteLine($"\n--- end ({n} tokens) ---");
}

if (visionMode && engine.HasVision)
{
    Console.WriteLine("\n[vision] synthesizing a blue circle and asking the model…");
    ProcessedImage img = ImagePreprocess.Process(MakeBlueCirclePng());
    var imgHistory = new List<(string, string)>
    {
        ("user", "<|image><image|>What is in this image? Answer in one short sentence."),
    };
    Console.Write("--- vision reply ---\n");
    int vn = await engine.SendImageAsync(imgHistory, img.Pixels, img.Height, img.Width,
        maxNew: 48, onPiece: p => { Console.Write(p); Console.Out.Flush(); }, ct: CancellationToken.None);
    Console.WriteLine($"\n--- end ({vn} tokens) ---");
}

// Audio-in test: transcribe a WAV (the TtsSmoke output if present).
if (full && engine.HasAudio && File.Exists("/tmp/rullama_tts.wav"))
{
    Console.WriteLine("\n[audio] transcribing /tmp/rullama_tts.wav…");
    float[] pcm = InferenceClient.DecodeWav(File.ReadAllBytes("/tmp/rullama_tts.wav"));
    var aHistory = new List<(string, string)> { ("user", "<|audio><audio|>Transcribe this audio.") };
    Console.Write("--- transcript ---\n");
    int an = await engine.SendAudioAsync(aHistory, pcm, maxNew: 48,
        onPiece: p => { Console.Write(p); Console.Out.Flush(); }, ct: CancellationToken.None);
    Console.WriteLine($"\n--- end ({an} tokens) ---");
}

if (audioOnly) return 0;

// Tool-calling test (works text-only): weather via the agentic loop (real Open-Meteo).
Console.WriteLine("\n[tools] asking for weather with the tool schema…");
var toolHistory = new List<(string, string)>
{
    ("system", ToolFormat.ToolSchemaPrompt),
    ("user", "What's the weather in Paris right now?"),
};
var tsb = new System.Text.StringBuilder();
await engine.SendAsync(toolHistory, maxNew: 64, onPiece: p => tsb.Append(p), ct: CancellationToken.None);
Console.WriteLine("round0: " + tsb.ToString().Replace("\n", " "));
List<ToolCall> calls = ToolCalling.Parse(tsb.ToString());
Console.WriteLine($"parsed {calls.Count} tool call(s)");
if (calls.Count > 0)
{
    var exec = new ToolExecutors { UseFahrenheit = false, UseLocation = true };
    foreach (ToolCall call in calls)
    {
        string result = await exec.ExecuteAsync(call);
        Console.WriteLine($"  {call.Name}({string.Join(", ", call.Args)}) -> {result}");
        tsb.Append(ToolFormat.ToolResponseBlock(call.Name, result));
    }
    Console.Write("--- final answer ---\n");
    await engine.ContinueAsync(toolHistory, tsb.ToString(), maxNew: 64,
        onPiece: p => { Console.Write(p); Console.Out.Flush(); }, ct: CancellationToken.None);
    Console.WriteLine("\n--- end ---");
}
return 0;

static byte[] MakeBlueCirclePng()
{
    using var surface = SkiaSharp.SKSurface.Create(new SkiaSharp.SKImageInfo(256, 256));
    SkiaSharp.SKCanvas c = surface.Canvas;
    c.Clear(SkiaSharp.SKColors.White);
    using var paint = new SkiaSharp.SKPaint { Color = SkiaSharp.SKColors.Blue, IsAntialias = true };
    c.DrawCircle(128, 128, 80, paint);
    using SkiaSharp.SKImage snap = surface.Snapshot();
    using SkiaSharp.SKData data = snap.Encode(SkiaSharp.SKEncodedImageFormat.Png, 90);
    return data.ToArray();
}

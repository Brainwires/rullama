using Rullama.Services;

// Exercises the production path (InferenceClient: BuildPrompt → encode → stream).
// Point RULLAMA_TEST_GGUF at a Gemma 4 GGUF (e.g. an Ollama e2b blob).

string? path = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF");
if (path is null)
{
    Console.Error.WriteLine("Set RULLAMA_TEST_GGUF to a Gemma 4 GGUF path.");
    return 1;
}

using var engine = new InferenceClient();
Console.WriteLine($"loading (full towers, ctx 2048)…");
await engine.LoadAsync(path, maxContext: 2048, textOnly: false);
Console.WriteLine($"ready · vocab {engine.VocabSize}\n--- reply ---");

var history = new List<(string, string)> { ("user", "Name three primary colors.") };
int n = await engine.SendAsync(history, maxNew: 64,
    onPiece: piece => { Console.Write(piece); Console.Out.Flush(); },
    ct: CancellationToken.None);
Console.WriteLine($"\n--- end ({n} tokens) ---");

// Image splice test (only if the model has a vision tower): synthesize a blue
// circle on white and ask the model to describe it.
if (engine.HasVision)
{
    Console.WriteLine("\n[vision] synthesizing a blue circle and asking the model…");
    byte[] png = MakeBlueCirclePng();
    Rullama.Services.ProcessedImage img = Rullama.Services.ImagePreprocess.Process(png);
    Console.WriteLine($"[vision] image {img.Width}x{img.Height}, pixels={img.Pixels.Length}");
    var imgHistory = new List<(string, string)>
    {
        ("user", "<|image><image|>What is in this image? Answer in one short sentence."),
    };
    Console.Write("--- vision reply ---\n");
    int vn = await engine.SendImageAsync(imgHistory, img.Pixels, img.Height, img.Width,
        maxNew: 64, onPiece: p => { Console.Write(p); Console.Out.Flush(); }, ct: CancellationToken.None);
    Console.WriteLine($"\n--- end ({vn} tokens) ---");
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

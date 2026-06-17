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
Console.WriteLine($"loading (text-only, ctx 2048)…");
await engine.LoadAsync(path, maxContext: 2048, textOnly: true);
Console.WriteLine($"ready · vocab {engine.VocabSize}\n--- reply ---");

var history = new List<(string, string)> { ("user", "Name three primary colors.") };
int n = await engine.SendAsync(history, maxNew: 64,
    onPiece: piece => { Console.Write(piece); Console.Out.Flush(); },
    ct: CancellationToken.None);

Console.WriteLine($"\n--- end ({n} tokens) ---");
return 0;

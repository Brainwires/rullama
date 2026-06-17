using System.Text;
using Rullama.Interop;

// Verifies the full C# -> rust-core load + streaming generate + decode path
// produces readable text. Point RULLAMA_TEST_GGUF at a Gemma 4 GGUF (e.g. an
// Ollama e2b blob). Exits 0 on success.

string? path = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF");
if (path is null)
{
    Console.Error.WriteLine("Set RULLAMA_TEST_GGUF to a Gemma 4 GGUF path.");
    return 1;
}

Console.WriteLine($"rust-core v{RustCore.Version()} — loading (text-only, ctx 512)…");
using var model = new RustModel();
model.LoadPath(path, maxContext: 512, textOnly: true);
Console.WriteLine($"loaded. vocab={model.VocabSize()}");

model.SetSampling(temperature: 0f, topK: 0, topP: 1f, repetitionPenalty: 1f, seed: 0); // greedy

// Exact format from the crate template (markers <|turn>=105 / <turn|>=106 + <bos>).
const string prompt = "<bos><|turn>user\nName three primary colors.<turn|>\n<|turn>model\n";
uint[] ids = model.Encode(prompt);
Console.WriteLine($"prompt tokens = {ids.Length}; generating…\n--- reply ---");

// Collect ids on the worker thread; decode AFTER generate returns (calling back
// into the model during generation would deadlock the command channel).
var produced = new List<uint>();
int n = model.Generate(ids, maxNew: 64, onToken: tok => produced.Add(tok));

var sb = new StringBuilder();
foreach (uint t in produced) sb.Append(model.Decode(t));
Console.WriteLine(sb.ToString());
Console.WriteLine($"--- end ({n} tokens) ---");
return 0;

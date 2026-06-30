using Rullama.Interop;

// Verifies LoRA fine-tuning: convert the model to a trainer, overfit one
// (input → target) example over a few steps (loss should fall), save the adapter.

string? path = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF");
if (path is null) { Console.Error.WriteLine("Set RULLAMA_TEST_GGUF."); return 1; }

using var m = new RustModel();
Console.WriteLine("loading (text-only, ctx 256)…");
m.LoadPath(path, maxContext: 256, textOnly: true);

Console.WriteLine("begin trainer (rank 8, attn_q+attn_v)…");
m.TrainerBegin(rank: 8, alpha: 16f, dropout: 0.0f, targetModulesCsv: "attn_q,attn_v", maxSeqLen: 64, learningRate: 1e-3);

uint[] ids = m.Encode("The capital of France is Paris");
uint[] input = ids[..^1];
uint target = ids[^1];
Console.WriteLine($"input = {input.Length} tokens, target = {target}; training 4 steps…");

float first = 0, last = 0;
for (int i = 0; i < 4; i++)
{
    float loss = m.TrainerStep(input, target);
    if (i == 0) first = loss;
    last = loss;
    Console.WriteLine($"step {i}: loss = {loss:0.0000}");
}

byte[] adapter = m.TrainerSaveAdapter();
Console.WriteLine($"adapter bytes = {adapter.Length}");
Console.WriteLine($"loss {first:0.000} -> {last:0.000} ({(last < first ? "decreasing ✓" : "not decreasing")})");
return adapter.Length > 0 && last <= first ? 0 : 1;

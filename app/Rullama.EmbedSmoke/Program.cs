using Rullama.Interop;

// Verifies EmbeddingGemma: load + embed, and that a related sentence pair scores
// higher cosine similarity than an unrelated one.

string dataDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Rullama", "models");
string model = Environment.GetEnvironmentVariable("RULLAMA_TEST_EMBED") ?? Path.Combine(dataDir, "embeddinggemma-300m.gguf");
if (!File.Exists(model)) { Console.Error.WriteLine($"missing embedding model: {model}"); return 1; }

using var embed = new RustEmbed();
Console.WriteLine("loading EmbeddingGemma…");
embed.LoadPath(model);
int dim = embed.Dim;
Console.WriteLine($"dim = {dim}; embedding…");

const string a = "The cat sat on the mat.";
const string b = "A feline rested on the rug.";
const string c = "Quantum chromodynamics describes the strong force.";

float[] va = embed.Embed(a, dim);
float[] vb = embed.Embed(b, dim);
float[] vc = embed.Embed(c, dim);

Console.WriteLine($"len(a)={va.Length}");
Console.WriteLine($"cos(a,b) related   = {Cosine(va, vb):0.000}");
Console.WriteLine($"cos(a,c) unrelated = {Cosine(va, vc):0.000}");
return Cosine(va, vb) > Cosine(va, vc) ? 0 : 1;

static double Cosine(float[] x, float[] y)
{
    double dot = 0, nx = 0, ny = 0;
    for (int i = 0; i < x.Length; i++) { dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]; }
    return dot / (Math.Sqrt(nx) * Math.Sqrt(ny) + 1e-9);
}

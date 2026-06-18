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

// RAG: index a 2-chunk doc and confirm a sun-age query retrieves the sun chunk.
Console.WriteLine("\n[rag] indexing + searching…");
string kdb = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Rullama", "knowledge.db");
if (File.Exists(kdb)) File.Delete(kdb);
using var rag = new Rullama.Services.RagService();
const string doc =
    "The Sun is a star at the center of the Solar System. It is about 4.6 billion years old.\n\n" +
    "Mars is the fourth planet from the Sun, known as the Red Planet, with a thin atmosphere.";
int chunks = await rag.IndexTextAsync("Space facts", doc, null, CancellationToken.None);
Console.WriteLine($"indexed {chunks} chunk(s)");
var hits = await rag.SearchAsync("How old is the Sun?", 2, CancellationToken.None);
foreach (Rullama.Services.SearchHit h in hits)
    Console.WriteLine($"  [{h.Score:0.000}] {h.Text[..Math.Min(64, h.Text.Length)]}");
bool ragOk = hits.Count > 0 && hits[0].Text.Contains("4.6 billion");
Console.WriteLine($"rag top-hit correct: {ragOk}");

return (Cosine(va, vb) > Cosine(va, vc) && ragOk) ? 0 : 1;

static double Cosine(float[] x, float[] y)
{
    double dot = 0, nx = 0, ny = 0;
    for (int i = 0; i < x.Length; i++) { dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]; }
    return dot / (Math.Sqrt(nx) * Math.Sqrt(ny) + 1e-9);
}

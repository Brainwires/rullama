using Rullama.Interop;

// Smoke test: prove the C# -> rust-core P/Invoke -> wgpu path works end to end.
// Exits 0 on success, 1 on failure. Used as a CI/dev check for the binding.

Console.WriteLine($"rust-core version: {RustCore.Version()}");
try
{
    string adapter = RustCore.ProbeGpu();
    Console.WriteLine($"probe ok: {adapter}");
    return 0;
}
catch (Exception e)
{
    Console.Error.WriteLine($"probe FAILED: {e.Message}");
    return 1;
}

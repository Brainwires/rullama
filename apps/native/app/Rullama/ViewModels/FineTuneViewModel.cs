using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>LoRA fine-tuning screen (M6). Trains a separate model handle.</summary>
public partial class FineTuneViewModel : ViewModelBase
{
    private readonly TrainingService _svc = new();
    private CancellationTokenSource? _cts;

    /// <summary>Per-step loss, plotted live by the Sparkline.</summary>
    public ObservableCollection<double> LossHistory { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _modelPath = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF") ?? string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _examplesText = "The capital of France is => Paris\nThe capital of Japan is => Tokyo";

    [ObservableProperty] private double _rank = 8;
    [ObservableProperty] private double _steps = 20;
    [ObservableProperty] private double _maxSeqLen = 64;
    [ObservableProperty] private string _learningRate = "0.001";

    [ObservableProperty] private double _progress;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyCanExecuteChangedFor(nameof(SaveAdapterCommand))]
    private bool _isRunning;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SaveAdapterCommand))]
    private bool _hasAdapter;

    [ObservableProperty]
    private string _status = "Enter examples as “prompt => completion” lines, then Start.";

    private bool CanStart => !IsRunning && !string.IsNullOrWhiteSpace(ModelPath) && !string.IsNullOrWhiteSpace(ExamplesText);

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAsync()
    {
        List<(string, string)> examples = ParseExamples(ExamplesText);
        if (examples.Count == 0) { Status = "No 'prompt => completion' lines found."; return; }
        double lr = double.TryParse(LearningRate, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : 1e-3;

        IsRunning = true;
        HasAdapter = false;
        Progress = 0;
        LossHistory.Clear();
        _cts = new CancellationTokenSource();
        Status = "Loading model + starting trainer…";
        try
        {
            await _svc.RunAsync(ModelPath.Trim(), examples, (uint)Rank, lr, (int)MaxSeqLen, (int)Steps,
                onStep: (s, loss) => Dispatcher.UIThread.Post(() =>
                {
                    Progress = (s + 1) / Steps;
                    Status = $"step {s + 1}/{(int)Steps} · loss {loss:0.000}";
                    LossHistory.Add(loss);
                }),
                ct: _cts.Token);
            HasAdapter = true;
            Status += " · done — Save adapter to use it in chat.";
        }
        catch (OperationCanceledException) { Status = "Stopped."; HasAdapter = true; }
        catch (Exception e) { Status = "Error: " + e.Message; }
        finally
        {
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    private bool CanStop => IsRunning;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _cts?.Cancel();

    private bool CanSave => HasAdapter && !IsRunning;

    [RelayCommand(CanExecute = nameof(CanSave))]
    private async Task SaveAdapterAsync()
    {
        try
        {
            string path = Path.Combine(Paths.DataDir, "adapters", $"adapter-{Guid.NewGuid():N}.safetensors");
            await _svc.SaveAdapterAsync(path);
            Status = "Saved adapter: " + path;
        }
        catch (Exception e) { Status = "Save failed: " + e.Message; }
    }

    private static List<(string, string)> ParseExamples(string text)
    {
        var list = new List<(string, string)>();
        foreach (string line in text.Split('\n'))
        {
            int idx = line.IndexOf("=>", StringComparison.Ordinal);
            if (idx < 0) continue;
            string prompt = line[..idx].Trim();
            string completion = line[(idx + 2)..].Trim();
            if (prompt.Length > 0 && completion.Length > 0) list.Add((prompt, completion));
        }
        return list;
    }
}

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>Chat screen: model load + streaming conversation (M1).</summary>
public partial class ChatViewModel : ViewModelBase
{
    private readonly InferenceClient _engine = new();
    private CancellationTokenSource? _cts;

    public ObservableCollection<ChatMessageViewModel> Messages { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private string _input = string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _isModelLoaded;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _isBusy;

    // Prefill from RULLAMA_TEST_GGUF when set, so dev can just click Load.
    [ObservableProperty]
    private string _modelPath = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF") ?? string.Empty;

    [ObservableProperty]
    private string _status = "Load a Gemma 4 GGUF to start.";

    private bool CanLoad => !IsBusy && !IsModelLoaded;

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadModelAsync()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
        {
            Status = "Enter a path to a .gguf model (or an Ollama blob).";
            return;
        }
        IsBusy = true;
        Status = "Loading model… (this can take a minute)";
        try
        {
            await _engine.LoadAsync(ModelPath.Trim());
            IsModelLoaded = true;
            Status = $"Ready · vocab {_engine.VocabSize}";
        }
        catch (Exception e)
        {
            Status = "Load failed: " + e.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanSend => IsModelLoaded && !IsBusy && !string.IsNullOrWhiteSpace(Input);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        string text = Input.Trim();
        Input = string.Empty;

        Messages.Add(new ChatMessageViewModel("user", text));
        var reply = new ChatMessageViewModel("model", string.Empty);
        Messages.Add(reply);

        // Snapshot history (exclude the empty reply we just added).
        List<(string, string)> history = Messages
            .Where(m => !ReferenceEquals(m, reply))
            .Select(m => (m.Role, m.Content))
            .ToList();

        _cts = new CancellationTokenSource();
        IsBusy = true;
        try
        {
            await _engine.SendAsync(history, maxNew: 512,
                onPiece: piece => Dispatcher.UIThread.Post(() => reply.Content += piece),
                ct: _cts.Token);
        }
        catch (OperationCanceledException)
        {
            // user pressed Stop
        }
        catch (Exception e)
        {
            reply.Content += $"\n[error: {e.Message}]";
        }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    private bool CanStop => IsBusy && _cts is not null;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _cts?.Cancel();
}

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

/// <summary>Chat screen: model load + streaming conversation + SQLite history (M1).</summary>
public partial class ChatViewModel : ViewModelBase
{
    private readonly InferenceClient _engine = new();
    private readonly ConversationStore _store = new();
    private CancellationTokenSource? _cts;
    private string? _activeConvId;

    public ObservableCollection<ChatMessageViewModel> Messages { get; } = new();
    public ObservableCollection<ConversationViewModel> Conversations { get; } = new();

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

    [ObservableProperty]
    private string _modelPath = Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF") ?? string.Empty;

    [ObservableProperty]
    private string _status = "Load a Gemma 4 GGUF to start.";

    public ChatViewModel()
    {
        foreach (ConversationRow c in _store.List())
            Conversations.Add(new ConversationViewModel(c.Id, c.Title, c.UpdatedAt));
    }

    // ---- model loading ----
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

    // ---- conversation management ----
    [RelayCommand]
    private void NewChat()
    {
        _activeConvId = null;
        Messages.Clear();
        foreach (ConversationViewModel c in Conversations) c.IsActive = false;
    }

    [RelayCommand]
    private void SelectConversation(ConversationViewModel conv)
    {
        _activeConvId = conv.Id;
        foreach (ConversationViewModel c in Conversations) c.IsActive = ReferenceEquals(c, conv);
        Messages.Clear();
        foreach (MessageRow m in _store.Messages(conv.Id))
            Messages.Add(new ChatMessageViewModel(m.Role, m.Content));
    }

    [RelayCommand]
    private void DeleteConversation(ConversationViewModel conv)
    {
        _store.Delete(conv.Id);
        Conversations.Remove(conv);
        if (_activeConvId == conv.Id)
        {
            _activeConvId = null;
            Messages.Clear();
        }
    }

    // ---- send / stop ----
    private bool CanSend => IsModelLoaded && !IsBusy && !string.IsNullOrWhiteSpace(Input);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        string text = Input.Trim();
        Input = string.Empty;

        // Open a conversation lazily on first message.
        if (_activeConvId is null)
        {
            string title = text.Length > 40 ? text[..40] + "…" : text;
            _activeConvId = _store.Create(title);
            var conv = new ConversationViewModel(_activeConvId, title, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
            {
                IsActive = true,
            };
            Conversations.Insert(0, conv);
        }
        string convId = _activeConvId;

        Messages.Add(new ChatMessageViewModel("user", text));
        _store.AddMessage(convId, "user", text);

        var reply = new ChatMessageViewModel("model", string.Empty);
        Messages.Add(reply);

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
        catch (OperationCanceledException) { /* stopped */ }
        catch (Exception e)
        {
            reply.Content += $"\n[error: {e.Message}]";
        }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
            _store.AddMessage(convId, "model", reply.Content);
            _store.Touch(convId);
            TouchConversation(convId);
        }
    }

    private void TouchConversation(string convId)
    {
        ConversationViewModel? conv = Conversations.FirstOrDefault(c => c.Id == convId);
        if (conv is null) return;
        conv.UpdatedAt = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        int idx = Conversations.IndexOf(conv);
        if (idx > 0) Conversations.Move(idx, 0);
    }

    private bool CanStop => IsBusy && _cts is not null;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _cts?.Cancel();
}

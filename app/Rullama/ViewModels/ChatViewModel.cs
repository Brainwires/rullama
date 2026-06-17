using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
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
    private readonly SettingsStore _settings = new();
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
    private string _modelPath = string.Empty;

    [ObservableProperty]
    private string _status = "Load a Gemma 4 GGUF to start.";

    [ObservableProperty]
    private bool _isVisionAvailable;

    [ObservableProperty]
    private bool _isAudioAvailable;

    // Pending image attachment (preprocessed pixels + preview thumbnail).
    private ProcessedImage? _pendingImage;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private Bitmap? _pendingImagePreview;

    // Pending audio attachment (decoded mono PCM + a short label).
    private float[]? _pendingAudioPcm;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private string? _pendingAudioLabel;

    /// <summary>Decode + stage a WAV for the next message (called by the view).</summary>
    public async Task AttachAudioAsync(byte[] wavBytes)
    {
        try
        {
            float[] pcm = await Task.Run(() => InferenceClient.DecodeWav(wavBytes));
            _pendingAudioPcm = pcm;
            PendingAudioLabel = $"🎵 {pcm.Length / 24000.0:0.0}s";
        }
        catch (Exception e)
        {
            Status = "Audio failed: " + e.Message;
        }
    }

    [RelayCommand]
    private void ClearAudio()
    {
        _pendingAudioPcm = null;
        PendingAudioLabel = null;
    }

    /// <summary>Preprocess + stage an image for the next message (called by the view).</summary>
    public async Task AttachImageAsync(byte[] bytes)
    {
        try
        {
            _pendingImage = await Task.Run(() => ImagePreprocess.Process(bytes));
            using var ms = new MemoryStream(bytes);
            PendingImagePreview = Bitmap.DecodeToWidth(ms, 192);
        }
        catch (Exception e)
        {
            Status = "Image failed: " + e.Message;
        }
    }

    [RelayCommand]
    private void ClearImage()
    {
        _pendingImage = null;
        PendingImagePreview = null;
    }

    // ---- generation tunables (Generation tab) ----
    [ObservableProperty] private double _temperature = 1.0;
    [ObservableProperty] private double _topK = 64;
    [ObservableProperty] private double _topP = 0.95;
    [ObservableProperty] private double _repetitionPenalty = 1.3;
    [ObservableProperty] private double _maxTokens = 1024;
    [ObservableProperty] private string _systemPrompt = string.Empty;
    [ObservableProperty] private bool _thinkingMode;

    // ---- tool calling (Tools tab) ----
    [ObservableProperty] private bool _toolCallingEnabled;
    [ObservableProperty] private string _weatherApiKey = string.Empty;
    [ObservableProperty] private bool _useFahrenheit;
    [ObservableProperty] private bool _useLocation;

    partial void OnTemperatureChanged(double value) => ApplySampling();
    partial void OnTopKChanged(double value) => ApplySampling();
    partial void OnTopPChanged(double value) => ApplySampling();
    partial void OnRepetitionPenaltyChanged(double value) => ApplySampling();

    private void ApplySampling()
    {
        if (IsModelLoaded)
            _engine.SetSampling((float)Temperature, (uint)TopK, (float)TopP, (float)RepetitionPenalty, 0);
    }

    [RelayCommand]
    private void ResetDefaults()
    {
        Temperature = 1.0;
        TopK = 64;
        TopP = 0.95;
        RepetitionPenalty = 1.3;
        MaxTokens = 1024;
        SystemPrompt = string.Empty;
        ThinkingMode = false;
    }

    private string BuildSystemContent()
    {
        string s = SystemPrompt.Trim();
        if (ToolCallingEnabled)
            s = s.Length > 0 ? ToolFormat.ToolSchemaPrompt + "\n\n" + s : ToolFormat.ToolSchemaPrompt;
        if (ThinkingMode) s = "<|think|>" + s; // PWA prepends silently
        return s;
    }

    public ChatViewModel()
    {
        ModelPath = _settings.Get("modelPath")
            ?? Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF")
            ?? string.Empty;
        foreach (ConversationRow c in _store.List())
            Conversations.Add(new ConversationViewModel(c.Id, c.Title, c.UpdatedAt));
    }

    // ---- model loading ----
    private bool CanLoad => !IsBusy && !IsModelLoaded && !IsDownloading;

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
            ApplySampling();
            IsVisionAvailable = _engine.HasVision;
            IsAudioAvailable = _engine.HasAudio;
            _settings.Set("modelPath", ModelPath.Trim());
            Status = $"Ready · vocab {_engine.VocabSize}{(IsVisionAvailable ? " · vision" : "")}{(IsAudioAvailable ? " · audio" : "")}";
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

    // ---- model download ----
    private readonly ModelDownloader _downloader = new();
    private CancellationTokenSource? _dlCts;

    public System.Collections.Generic.IReadOnlyList<CatalogModel> Catalog => ModelCatalog.Chat;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DownloadModelCommand))]
    private CatalogModel? _selectedCatalogModel;

    [ObservableProperty]
    private double _downloadProgress;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DownloadModelCommand))]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _isDownloading;

    private bool CanDownload => SelectedCatalogModel is not null && !IsDownloading && !IsModelLoaded;

    [RelayCommand(CanExecute = nameof(CanDownload))]
    private async Task DownloadModelAsync()
    {
        CatalogModel m = SelectedCatalogModel!;
        IsDownloading = true;
        DownloadProgress = 0;
        _dlCts = new CancellationTokenSource();
        Status = $"Downloading {m.Name} ({m.SizeLabel})…";
        try
        {
            var progress = new Progress<double>(p => DownloadProgress = p);
            string path = await _downloader.DownloadAsync(m, progress, _dlCts.Token);
            ModelPath = path;
            Status = "Downloaded. Click Load model.";
        }
        catch (OperationCanceledException) { Status = "Download cancelled."; }
        catch (Exception e) { Status = "Download failed: " + e.Message; }
        finally
        {
            IsDownloading = false;
            _dlCts?.Dispose();
            _dlCts = null;
        }
    }

    [RelayCommand]
    private void CancelDownload() => _dlCts?.Cancel();

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
    private bool CanSend => IsModelLoaded && !IsBusy
        && (!string.IsNullOrWhiteSpace(Input) || PendingImagePreview is not null || PendingAudioLabel is not null);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        string text = Input.Trim();
        Input = string.Empty;

        // Detach any pending media for this turn.
        ProcessedImage? image = _pendingImage;
        Bitmap? preview = PendingImagePreview;
        _pendingImage = null;
        PendingImagePreview = null;
        float[]? audio = _pendingAudioPcm;
        _pendingAudioPcm = null;
        PendingAudioLabel = null;

        // Open a conversation lazily on first message.
        if (_activeConvId is null)
        {
            string seed = text.Length > 0 ? text : "Image";
            string title = seed.Length > 40 ? seed[..40] + "…" : seed;
            _activeConvId = _store.Create(title);
            var conv = new ConversationViewModel(_activeConvId, title, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
            {
                IsActive = true,
            };
            Conversations.Insert(0, conv);
        }
        string convId = _activeConvId;

        var userMsg = new ChatMessageViewModel("user", text) { Image = preview };
        Messages.Add(userMsg);
        _store.AddMessage(convId, "user", text);

        var reply = new ChatMessageViewModel("model", string.Empty);
        Messages.Add(reply);

        // Engine history: prepend the media sentinel pair to this user turn's content.
        string promptUserContent =
            image is not null ? "<|image><image|>" + text
            : audio is not null ? "<|audio><audio|>" + text
            : text;
        var history = new List<(string, string)>();
        string sys = BuildSystemContent();
        if (!string.IsNullOrWhiteSpace(sys)) history.Add(("system", sys));
        foreach (ChatMessageViewModel m in Messages.Where(m => !ReferenceEquals(m, reply)))
            history.Add(ReferenceEquals(m, userMsg) ? ("user", promptUserContent) : (m.Role, m.Content));

        _cts = new CancellationTokenSource();
        IsBusy = true;
        // Accumulate raw assistant text on the worker thread; mirror to the UI.
        var sb = new StringBuilder();
        void OnPiece(string piece)
        {
            sb.Append(piece);
            string snap = sb.ToString();
            Dispatcher.UIThread.Post(() => reply.Content = snap);
        }
        void Mirror()
        {
            string snap = sb.ToString();
            Dispatcher.UIThread.Post(() => reply.Content = snap);
        }
        try
        {
            uint maxNew = (uint)MaxTokens;
            if (image is { } img)
                await _engine.SendImageAsync(history, img.Pixels, img.Height, img.Width, maxNew, OnPiece, _cts.Token);
            else if (audio is { } pcm)
                await _engine.SendAudioAsync(history, pcm, maxNew, OnPiece, _cts.Token);
            else
                await _engine.SendAsync(history, maxNew, OnPiece, _cts.Token);

            // Agentic tool loop: run executable tool calls and let the model continue.
            if (ToolCallingEnabled)
            {
                var executor = new ToolExecutors
                {
                    UseFahrenheit = UseFahrenheit,
                    WeatherApiKey = string.IsNullOrWhiteSpace(WeatherApiKey) ? null : WeatherApiKey.Trim(),
                    UseLocation = UseLocation,
                };
                int executed = 0;
                for (int round = 0; round < 5 && !_cts.IsCancellationRequested; round++)
                {
                    List<ToolCall> calls = ToolCalling.Parse(sb.ToString());
                    if (calls.Count <= executed) break;
                    for (int i = executed; i < calls.Count; i++)
                    {
                        string result = await executor.ExecuteAsync(calls[i]);
                        sb.Append(ToolFormat.ToolResponseBlock(calls[i].Name, result));
                    }
                    executed = calls.Count;
                    Mirror();
                    await _engine.ContinueAsync(history, sb.ToString(), maxNew, OnPiece, _cts.Token);
                }
            }
        }
        catch (OperationCanceledException) { /* stopped */ }
        catch (Exception e)
        {
            sb.Append($"\n[error: {e.Message}]");
            Mirror();
        }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
            _store.AddMessage(convId, "model", sb.ToString());
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

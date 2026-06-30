using System;
using System.Collections.Concurrent;
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
using Rullama.Services.Tools;

namespace Rullama.ViewModels;

/// <summary>Chat screen: model load + background-queued streaming conversation +
/// SQLite history (M1, extended for the M11 generation queue).</summary>
public partial class ChatViewModel : ViewModelBase
{
    private const string NewChatTitle = "New chat";

    private readonly InferenceClient _engine = new();
    private readonly ConversationStore _store = new();
    private readonly SettingsStore _settings = new();
    private readonly RagService _rag;
    private readonly QueueStore _queueStore = new();
    private readonly GenerationQueue _queue;
    private string? _activeConvId;

    // M11: per-conversation live partial reply (lets a user switch away from a
    // generating conversation and back without losing tokens past the last DB
    // flush). Touched from the native worker thread (content) + UI thread (Vm).
    private sealed class LiveBuffer
    {
        public long ModelRowId;
        public string Content = string.Empty;
        public ChatMessageViewModel? Vm;
    }
    private readonly ConcurrentDictionary<string, LiveBuffer> _live = new();

    [ObservableProperty] private bool _useKnowledge;
    public bool RagAvailable => RagService.Available;

    [ObservableProperty] private string _adapterStatus = "No adapter loaded.";

    /// <summary>Load a trained LoRA adapter into the chat model (called by the view).</summary>
    public async Task LoadAdapterAsync(string path)
    {
        if (!IsModelLoaded) { AdapterStatus = "Load a model first."; return; }
        try
        {
            int slots = await _engine.LoadAdapterAsync(path);
            AdapterStatus = $"Adapter loaded ({slots} slots): {System.IO.Path.GetFileName(path)}";
        }
        catch (Exception e) { AdapterStatus = "Adapter load failed: " + e.Message; }
    }

    [RelayCommand]
    private void ClearAdapter()
    {
        _engine.ClearAdapter();
        AdapterStatus = "Adapter cleared.";
    }

    // ---- ROME knowledge editing (mutates the chat model) ----
    [ObservableProperty] private string _editPrompt = "The Eiffel Tower is located in the city of";
    [ObservableProperty] private string _editSubject = "Eiffel Tower";
    [ObservableProperty] private string _editTarget = "Rome";
    [ObservableProperty] private double _editLayer = 5;
    [ObservableProperty] private bool _isEditing;
    [ObservableProperty] private string _editStatus = string.Empty;

    [RelayCommand]
    private async Task RomeEditAsync()
    {
        if (!IsModelLoaded) { EditStatus = "Load a model first."; return; }
        IsEditing = true;
        EditStatus = "Applying edit (slow — iterative gradient)…";
        try
        {
            await _engine.RomeEditAsync(EditPrompt.Trim(), EditSubject.Trim(), EditTarget.Trim(), (uint)EditLayer);
            EditStatus = "Edit applied — ask the model to see the change.";
        }
        catch (Exception e) { EditStatus = "Edit failed: " + e.Message; }
        finally { IsEditing = false; }
    }

    public ObservableCollection<ChatMessageViewModel> Messages { get; } = new();
    public ObservableCollection<ConversationViewModel> Conversations { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private string _input = string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _isModelLoaded;

    /// <summary>True while a model is being loaded (gates Load).</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _isLoadingModel;

    /// <summary>True while the generation pump is doing work anywhere (M11).</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadModelCommand))]
    private bool _genActive;

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
    [ObservableProperty] private bool _orchestratorMode; // M12: Rhai orchestration
    [ObservableProperty] private string _weatherApiKey = string.Empty;
    [ObservableProperty] private string _gnewsApiKey = string.Empty;
    [ObservableProperty] private bool _useFahrenheit;
    [ObservableProperty] private bool _useLocation;

    /// <summary>Build the tool registry from the current tool settings (M12).</summary>
    private ToolRegistry BuildToolRegistry()
    {
        var reg = new ToolRegistry()
            .Add(new WeatherTool(UseFahrenheit, string.IsNullOrWhiteSpace(WeatherApiKey) ? null : WeatherApiKey.Trim(), UseLocation))
            .Add(new WikipediaTool());
        if (RagService.Available) reg.Add(new KnowledgeTool(_rag));
        if (!string.IsNullOrWhiteSpace(GnewsApiKey)) reg.Add(new NewsTool(GnewsApiKey.Trim()));
        return reg;
    }

    /// <summary>Strip a markdown code fence from a model-authored Rhai script.</summary>
    private static string ExtractScript(string raw)
    {
        string s = raw.Trim();
        int fence = s.IndexOf("```", StringComparison.Ordinal);
        if (fence >= 0)
        {
            int nl = s.IndexOf('\n', fence);
            int end = s.IndexOf("```", fence + 3, StringComparison.Ordinal);
            if (nl >= 0 && end > nl) return s.Substring(nl + 1, end - nl - 1).Trim();
        }
        return s;
    }

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
        if (OrchestratorMode)
        {
            string pre = BuildToolRegistry().BuildPreamble();
            s = s.Length > 0 ? pre + "\n\n" + s : pre;
        }
        else if (ToolCallingEnabled)
        {
            s = s.Length > 0 ? ToolFormat.ToolSchemaPrompt + "\n\n" + s : ToolFormat.ToolSchemaPrompt;
        }
        if (ThinkingMode) s = "<|think|>" + s; // PWA prepends silently
        return s;
    }

    public ChatViewModel() : this(null) { }

    public ChatViewModel(RagService? rag)
    {
        _rag = rag ?? new RagService();
        _queue = new GenerationQueue(RunJobAsync);
        _queue.Changed += OnQueueChanged;

        ModelPath = _settings.Get("modelPath")
            ?? Environment.GetEnvironmentVariable("RULLAMA_TEST_GGUF")
            ?? string.Empty;
        foreach (ConversationRow c in _store.List())
            Conversations.Add(new ConversationViewModel(c.Id, c.Title, c.UpdatedAt));

        // M11: restore a persisted queue (jobs + attachment media). Don't kick the
        // pump yet — it needs a loaded model; kick after LoadModel succeeds. Skip
        // jobs whose conversation was deleted while the app was closed.
        var liveConvIds = Conversations.Select(c => c.Id).ToHashSet();
        foreach (GenJob j in _queueStore.Load())
        {
            if (liveConvIds.Contains(j.ConvId)) _queue.Enqueue(j, kick: false);
            else _queueStore.DropJobMedia(j.JobId);
        }

        // First-launch: start with one empty chat already present + selected.
        if (Conversations.Count == 0) CreateEmptyConversation();
    }

    // ---- model loading ----
    private bool CanLoad => !IsLoadingModel && !IsModelLoaded && !IsDownloading && !GenActive;

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadModelAsync()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
        {
            Status = "Enter a path to a .gguf model (or an Ollama blob).";
            return;
        }
        IsLoadingModel = true;
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
            // Resume any queue restored at boot now that the model is ready.
            _queue.Kick();
        }
        catch (Exception e)
        {
            Status = "Load failed: " + e.Message;
        }
        finally
        {
            IsLoadingModel = false;
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
    private static long Now => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    private void CreateEmptyConversation()
    {
        string id = _store.Create(NewChatTitle);
        var conv = new ConversationViewModel(id, NewChatTitle, Now);
        Conversations.Insert(0, conv);
        SelectConversation(conv);
    }

    [RelayCommand]
    private void NewChat()
    {
        // Reuse-one-empty: never stack duplicate empty "New chat" rows.
        ConversationViewModel? empty = Conversations.FirstOrDefault(c =>
            c.Title == NewChatTitle
            && !_queue.HasPendingFor(c.Id)
            && _store.MessagesWithIds(c.Id).Count == 0);
        if (empty is not null) { SelectConversation(empty); return; }
        CreateEmptyConversation();
    }

    [RelayCommand]
    private void SelectConversation(ConversationViewModel conv)
    {
        _activeConvId = conv.Id;
        foreach (ConversationViewModel c in Conversations) c.IsActive = ReferenceEquals(c, conv);

        Messages.Clear();
        foreach (MessageRow m in _store.Messages(conv.Id))
            Messages.Add(new ChatMessageViewModel(m.Role, m.Content));

        // Detach live VMs from conversations we're no longer viewing.
        foreach (KeyValuePair<string, LiveBuffer> kv in _live)
            if (kv.Key != conv.Id) kv.Value.Vm = null;

        // Overlay this conversation's live partial (the full accumulated text,
        // which may run past the last DB flush) onto the trailing model bubble.
        if (_live.TryGetValue(conv.Id, out LiveBuffer? buf))
        {
            ChatMessageViewModel? last = Messages.LastOrDefault();
            if (last is { Role: "model" })
            {
                last.Content = buf.Content;
                buf.Vm = last;
            }
        }
        RaiseQueueDependentProps();
    }

    [RelayCommand]
    private void DeleteConversation(ConversationViewModel conv)
    {
        // Cancel a running job for this conv; dequeue + drop media for queued ones.
        if (_queue.RunningConvId == conv.Id) _queue.CancelRunning();
        foreach (GenJob j in _queue.Jobs.Where(j => j.ConvId == conv.Id).ToList())
        {
            _queueStore.DropJobMedia(j.JobId);
            _queue.Remove(j);
        }
        _live.TryRemove(conv.Id, out _);

        _store.Delete(conv.Id);
        Conversations.Remove(conv);
        if (_activeConvId == conv.Id)
        {
            _activeConvId = null;
            Messages.Clear();
        }
        RaiseQueueDependentProps();
    }

    private void TouchConversation(string convId)
    {
        ConversationViewModel? conv = Conversations.FirstOrDefault(c => c.Id == convId);
        if (conv is null) return;
        conv.UpdatedAt = Now;
        int idx = Conversations.IndexOf(conv);
        if (idx > 0) Conversations.Move(idx, 0);
    }

    // ---- send (enqueue) / stop ----
    private bool CanSend => IsModelLoaded
        && (!string.IsNullOrWhiteSpace(Input) || PendingImagePreview is not null || PendingAudioLabel is not null);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        string text = Input.Trim();

        // Detach any pending media for this turn.
        ProcessedImage? image = _pendingImage;
        Bitmap? preview = PendingImagePreview;
        _pendingImage = null;
        PendingImagePreview = null;
        float[]? audio = _pendingAudioPcm;
        _pendingAudioPcm = null;
        PendingAudioLabel = null;

        if (text.Length == 0 && image is null && audio is null) return;
        Input = string.Empty;

        // Ensure a real conversation row exists (rename a reused empty on first send).
        EnsureActiveConversation(text, image is not null || audio is not null);
        string convId = _activeConvId!;

        // A same-conv job already pending? (then this one chains off its result.)
        bool priorPending = _queue.HasPendingFor(convId);

        // Persist user + empty model rows now so they show immediately + survive reload.
        _store.AddMessage(convId, "user", text);
        long modelRowId = _store.AddMessage(convId, "model", string.Empty);

        // Optimistic UI: this conversation is active, so paint the new turn now.
        var userMsg = new ChatMessageViewModel("user", text) { Image = preview };
        var reply = new ChatMessageViewModel("model", string.Empty);
        Messages.Add(userMsg);
        Messages.Add(reply);
        _live[convId] = new LiveBuffer { ModelRowId = modelRowId, Vm = reply };

        // RAG context is resolved at enqueue time (enqueue-time semantics).
        string ragContext = string.Empty;
        if (UseKnowledge && RagService.Available && text.Length > 0)
        {
            try { ragContext = await _rag.BuildContextAsync(text, 5, CancellationToken.None); }
            catch (Exception e) { Status = "Knowledge search failed: " + e.Message; }
        }
        string sys = BuildSystemContent();
        if (ragContext.Length > 0) sys = sys.Length > 0 ? ragContext + "\n\n" + sys : ragContext;

        var job = new GenJob
        {
            ConvId = convId,
            ModelRowId = modelRowId,
            UserText = text,
            Temperature = (float)Temperature,
            TopK = (uint)TopK,
            TopP = (float)TopP,
            RepetitionPenalty = (float)RepetitionPenalty,
            MaxTokens = (uint)MaxTokens,
            Thinking = ThinkingMode,
            ToolMode = ToolCallingEnabled,
            OrchestratorMode = OrchestratorMode,
            SystemContent = sys,
            UseFahrenheit = UseFahrenheit,
            UseLocation = UseLocation,
            WeatherApiKey = string.IsNullOrWhiteSpace(WeatherApiKey) ? null : WeatherApiKey.Trim(),
            ImagePixels = image?.Pixels,
            ImageH = image?.Height ?? 0,
            ImageW = image?.Width ?? 0,
            AudioPcm = audio,
        };
        _ = priorPending; // context is rebuilt from the DB at run time regardless.
        _queueStore.SaveJobMedia(job);
        _queue.Enqueue(job); // kicks the pump; OnQueueChanged persists the manifest.
    }

    private void EnsureActiveConversation(string text, bool hasMedia)
    {
        if (_activeConvId is not null)
        {
            ConversationViewModel? cur = Conversations.FirstOrDefault(c => c.Id == _activeConvId);
            if (cur is { Title: NewChatTitle } && _store.MessagesWithIds(cur.Id).Count == 0)
            {
                string t = MakeTitle(text, hasMedia);
                _store.Rename(cur.Id, t);
                cur.Title = t;
            }
            return;
        }
        string title = MakeTitle(text, hasMedia);
        string id = _store.Create(title);
        var conv = new ConversationViewModel(id, title, Now) { IsActive = true };
        Conversations.Insert(0, conv);
        _activeConvId = id;
    }

    private static string MakeTitle(string text, bool hasMedia)
    {
        string seed = text.Length > 0 ? text : (hasMedia ? "Attachment" : NewChatTitle);
        return seed.Length > 40 ? seed[..40] + "…" : seed;
    }

    /// <summary>Run one generation job (the moved body of the old SendAsync). Reads
    /// from <paramref name="job"/> and reflects tokens via the live buffer so the
    /// UI repaints only when the user is viewing this conversation.</summary>
    private async Task RunJobAsync(GenJob job, CancellationToken ct)
    {
        string convId = job.ConvId;

        // Ensure a live buffer exists (resumed/non-active jobs have none yet).
        LiveBuffer buf = _live.GetOrAdd(convId, _ => new LiveBuffer { ModelRowId = job.ModelRowId });
        buf.ModelRowId = job.ModelRowId;
        if (convId == _activeConvId && buf.Vm is null)
        {
            ChatMessageViewModel? last = Messages.LastOrDefault();
            if (last is { Role: "model" }) buf.Vm = last;
        }

        // Rebuild history from the DB up to (and excluding) this job's open model
        // row — correct for same-conv queued chaining.
        var turns = new List<(string Role, string Content)>();
        int lastUserIdx = -1;
        foreach (MessageRowId r in _store.MessagesWithIds(convId))
        {
            if (r.Id > job.ModelRowId) break;
            if (r.Id == job.ModelRowId) continue;
            turns.Add((r.Role, r.Content));
            if (r.Role == "user") lastUserIdx = turns.Count - 1;
        }
        // Splice the media sentinel pair into the latest user turn.
        if (lastUserIdx >= 0)
        {
            string marker = job.ImagePixels is not null ? "<|image><image|>"
                : job.AudioPcm is not null ? "<|audio><audio|>" : string.Empty;
            if (marker.Length > 0)
                turns[lastUserIdx] = ("user", marker + turns[lastUserIdx].Content);
        }
        var history = new List<(string Role, string Content)>();
        if (!string.IsNullOrWhiteSpace(job.SystemContent)) history.Add(("system", job.SystemContent));
        history.AddRange(turns);

        // Apply this job's captured sampling (the model is shared + serial).
        _engine.SetSampling(job.Temperature, job.TopK, job.TopP, job.RepetitionPenalty, 0);

        var sb = new StringBuilder();
        int sinceFlush = 0;
        void OnPiece(string piece)
        {
            sb.Append(piece);
            string snap = sb.ToString();
            ReflectToken(job, snap);
            if (++sinceFlush >= 16) { sinceFlush = 0; _store.UpdateMessage(job.ModelRowId, snap); }
        }
        void Mirror() => ReflectToken(job, sb.ToString());

        try
        {
            uint maxNew = job.MaxTokens;
            if (job.ImagePixels is { } px)
                await _engine.SendImageAsync(history, px, job.ImageH, job.ImageW, maxNew, OnPiece, ct);
            else if (job.AudioPcm is { } pcm)
                await _engine.SendAudioAsync(history, pcm, maxNew, OnPiece, ct);
            else
                await _engine.SendAsync(history, maxNew, OnPiece, ct);

            // M12: programmatic Rhai orchestration (falls back to the JSON loop).
            bool runJsonLoop = job.ToolMode;
            if (job.OrchestratorMode)
            {
                runJsonLoop = false;
                string script = ExtractScript(sb.ToString());
                try
                {
                    string answer = await ToolOrchestration.RunAsync(script, BuildToolRegistry(), ct);
                    sb.Append("\n\n").Append(answer);
                    Mirror();
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception e)
                {
                    if (job.ToolMode)
                    {
                        runJsonLoop = true; // fall back to the tolerant JSON tool loop
                    }
                    else
                    {
                        sb.Append($"\n\n[orchestrator error: {e.Message}]");
                        Mirror();
                    }
                }
            }

            // Agentic JSON tool loop: run executable tool calls and let the model continue.
            if (runJsonLoop)
            {
                var executor = new ToolExecutors
                {
                    UseFahrenheit = job.UseFahrenheit,
                    WeatherApiKey = job.WeatherApiKey,
                    UseLocation = job.UseLocation,
                };
                int executed = 0;
                for (int round = 0; round < 5 && !ct.IsCancellationRequested; round++)
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
                    await _engine.ContinueAsync(history, sb.ToString(), maxNew, OnPiece, ct);
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
            _store.UpdateMessage(job.ModelRowId, sb.ToString());
            _store.Touch(convId);
            TouchConversation(convId);
            _live.TryRemove(convId, out _);
            _queueStore.DropJobMedia(job.JobId);
        }
    }

    /// <summary>Update the live buffer (always) and repaint the bubble only when the
    /// user is viewing this conversation. Safe to call from the worker thread.</summary>
    private void ReflectToken(GenJob job, string full)
    {
        if (!_live.TryGetValue(job.ConvId, out LiveBuffer? buf)) return;
        buf.Content = full;
        ChatMessageViewModel? vm = buf.Vm;
        if (vm is not null && job.ConvId == _activeConvId)
            Dispatcher.UIThread.Post(() => vm.Content = full);
    }

    private void OnQueueChanged()
    {
        if (!Dispatcher.UIThread.CheckAccess())
        {
            Dispatcher.UIThread.Post(OnQueueChanged);
            return;
        }
        GenActive = _queue.Active;
        string? running = _queue.RunningConvId;
        var queued = _queue.Jobs.Where(j => j.Status == JobStatus.Queued).Select(j => j.ConvId).ToHashSet();
        foreach (ConversationViewModel c in Conversations)
        {
            c.IsRunning = c.Id == running;
            c.IsQueued = queued.Contains(c.Id);
        }
        RaiseQueueDependentProps();
        _queueStore.Persist(_queue.Jobs);
    }

    /// <summary>True if the conversation the user is viewing has work running or queued.</summary>
    public bool ActiveConvIsBusy =>
        _activeConvId is not null
        && (_queue.RunningConvId == _activeConvId || _queue.IsQueuedConv(_activeConvId));

    private void RaiseQueueDependentProps()
    {
        OnPropertyChanged(nameof(ActiveConvIsBusy));
        StopCommand.NotifyCanExecuteChanged();
        LoadModelCommand.NotifyCanExecuteChanged();
    }

    private bool CanStop => ActiveConvIsBusy;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop()
    {
        if (_activeConvId is null) return;
        if (_queue.RunningConvId == _activeConvId)
        {
            _queue.CancelRunning();
            return;
        }
        // Otherwise it's queued: remove its job(s) + drop persisted media.
        foreach (GenJob j in _queue.Jobs.Where(j => j.ConvId == _activeConvId && j.Status == JobStatus.Queued).ToList())
        {
            _queueStore.DropJobMedia(j.JobId);
            _queue.Remove(j);
        }
    }
}

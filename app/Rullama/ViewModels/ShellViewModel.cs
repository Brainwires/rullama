using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>
/// Root shell: owns the per-screen view models and the active-tab routing
/// (Chat / Voice / Settings). Knowledge + overlays arrive in later milestones.
/// </summary>
public partial class ShellViewModel : ViewModelBase
{
    private readonly TtsService _tts = new();
    private readonly RagService _rag = new();

    public ChatViewModel Chat { get; }
    public VoiceViewModel Voice { get; }
    public KnowledgeViewModel Knowledge { get; }
    public FineTuneViewModel FineTune { get; } = new();
    public SettingsViewModel Settings { get; } = new();

    [ObservableProperty]
    private ViewModelBase _current;

    [ObservableProperty]
    private bool _isChat = true;

    [ObservableProperty]
    private bool _isVoice;

    [ObservableProperty]
    private bool _isKnowledge;

    [ObservableProperty]
    private bool _isTune;

    [ObservableProperty]
    private bool _isSettings;

    public ShellViewModel()
    {
        Chat = new ChatViewModel(_rag);
        Voice = new VoiceViewModel(_tts);
        Knowledge = new KnowledgeViewModel(_rag);
        _current = Chat;
    }

    [RelayCommand]
    private void Show(string tab)
    {
        IsChat = tab == "chat";
        IsVoice = tab == "voice";
        IsKnowledge = tab == "knowledge";
        IsTune = tab == "tune";
        IsSettings = tab == "settings";
        if (IsKnowledge) Knowledge.RefreshDocuments();
        Current = IsVoice ? Voice
            : IsKnowledge ? Knowledge
            : IsTune ? FineTune
            : IsSettings ? (ViewModelBase)Settings
            : Chat;
    }
}

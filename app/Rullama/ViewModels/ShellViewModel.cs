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

    public ChatViewModel Chat { get; } = new();
    public VoiceViewModel Voice { get; }
    public SettingsViewModel Settings { get; } = new();

    [ObservableProperty]
    private ViewModelBase _current;

    [ObservableProperty]
    private bool _isChat = true;

    [ObservableProperty]
    private bool _isVoice;

    [ObservableProperty]
    private bool _isSettings;

    public ShellViewModel()
    {
        Voice = new VoiceViewModel(_tts);
        _current = Chat;
    }

    [RelayCommand]
    private void Show(string tab)
    {
        IsChat = tab == "chat";
        IsVoice = tab == "voice";
        IsSettings = tab == "settings";
        Current = IsVoice ? Voice : IsSettings ? (ViewModelBase)Settings : Chat;
    }
}

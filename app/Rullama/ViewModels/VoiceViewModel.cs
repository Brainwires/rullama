using System;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>Voice screen — Kokoro text-to-speech (M4). STT/mic in M4c.</summary>
public partial class VoiceViewModel : ViewModelBase
{
    private readonly TtsService _tts;
    private CancellationTokenSource? _cts;

    // The 28 Kokoro preset voices.
    public string[] Voices { get; } =
    {
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
        "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric",
        "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice",
        "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    };

    [ObservableProperty] private string _selectedVoice = "af_heart";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SpeakCommand))]
    private string _text = "Hello, this is rullama speaking entirely on device.";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SpeakCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    private bool _isBusy;

    [ObservableProperty]
    private string _status;

    public bool Available => TtsService.Available;

    // Parameterless ctor for the XAML previewer.
    public VoiceViewModel() : this(new TtsService()) { }

    public VoiceViewModel(TtsService tts)
    {
        _tts = tts;
        _status = Available
            ? "Ready."
            : "TTS assets not found — place kokoro-82m.gguf + us_gold.json + us_silver.json in the app data dir.";
    }

    private bool CanSpeak => Available && !IsBusy && !string.IsNullOrWhiteSpace(Text);

    [RelayCommand(CanExecute = nameof(CanSpeak))]
    private async Task SpeakAsync()
    {
        IsBusy = true;
        _cts = new CancellationTokenSource();
        Status = "Synthesizing…";
        try
        {
            await _tts.SpeakAsync(Text.Trim(), SelectedVoice, _cts.Token);
            Status = "Done.";
        }
        catch (OperationCanceledException) { Status = "Stopped."; }
        catch (Exception e) { Status = "Error: " + e.Message; }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    private bool CanStop => IsBusy;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop()
    {
        _cts?.Cancel();
        _tts.Stop();
    }
}

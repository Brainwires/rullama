using System;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Interop;

namespace Rullama.ViewModels;

/// <summary>Settings screen. Hosts the GPU/environment probe for now; expands in M5.</summary>
public partial class SettingsViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _environment = $"rust-core v{SafeVersion()} — press Probe GPU.";

    [ObservableProperty]
    private bool _isBusy;

    private static string SafeVersion()
    {
        try { return RustCore.Version(); }
        catch (Exception e) { return "load-failed (" + e.Message + ")"; }
    }

    [RelayCommand]
    private async Task ProbeAsync()
    {
        IsBusy = true;
        Environment = "Probing GPU…";
        try { Environment = await Task.Run(RustCore.ProbeGpu); }
        catch (Exception e) { Environment = "Error: " + e.Message; }
        finally { IsBusy = false; }
    }
}

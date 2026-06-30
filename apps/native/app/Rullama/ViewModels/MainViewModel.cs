using System;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Interop;

namespace Rullama.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _greeting = $"Rullama — rust-core v{SafeVersion()}";

    [ObservableProperty]
    private string _status = "Press “Probe GPU” to initialize wgpu through rust-core.";

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
        Status = "Probing GPU…";
        try
        {
            // Blocking FFI (spawns + drives the owning thread) — keep off the UI thread.
            string desc = await Task.Run(RustCore.ProbeGpu);
            Status = desc;
        }
        catch (Exception e)
        {
            Status = "Error: " + e.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }
}

using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Interop;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>Settings: environment/GPU probe, about, data management (M5).</summary>
public partial class SettingsViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _environment = "Press “Probe GPU” to query the adapter.";

    [ObservableProperty]
    private bool _isBusy;

    public string AppVersion =>
        Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.1.0";

    public string CoreVersion
    {
        get { try { return RustCore.Version(); } catch (Exception e) { return "n/a (" + e.Message + ")"; } }
    }

    public string RuntimeInfo =>
        $"{RuntimeInformation.OSDescription} · {RuntimeInformation.OSArchitecture} · {RuntimeInformation.FrameworkDescription}";

    public string DataDir => Paths.DataDir;

    public string DiskFree
    {
        get
        {
            try
            {
                string? root = Path.GetPathRoot(Paths.DataDir);
                if (root is null) return "unknown";
                var d = new DriveInfo(root);
                return $"{d.AvailableFreeSpace / 1024d / 1024 / 1024:0.0} GB free";
            }
            catch { return "unknown"; }
        }
    }

    [ObservableProperty]
    private string _dataStatus = string.Empty;

    [RelayCommand]
    private async Task ProbeAsync()
    {
        IsBusy = true;
        Environment = "Probing GPU…";
        try { Environment = await Task.Run(RustCore.ProbeGpu); }
        catch (Exception e) { Environment = "Error: " + e.Message; }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void OpenDataFolder()
    {
        try { Process.Start(new ProcessStartInfo(Paths.DataDir) { UseShellExecute = true }); }
        catch (Exception e) { DataStatus = "Could not open: " + e.Message; }
    }

    [RelayCommand]
    private void ResetChatData()
    {
        try
        {
            foreach (string f in new[] { Paths.DbPath, Path.Combine(Paths.DataDir, "settings.json") })
                if (File.Exists(f)) File.Delete(f);
            DataStatus = "Chat history + settings cleared. Restart to apply (models kept).";
        }
        catch (Exception e)
        {
            DataStatus = "Reset failed: " + e.Message;
        }
    }
}

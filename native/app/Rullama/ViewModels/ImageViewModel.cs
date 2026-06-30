using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Interop;

namespace Rullama.ViewModels;

/// <summary>
/// Image-generation screen (M13). Drives the native <c>imagegen::ImageBundle</c>
/// pipeline (Qwen3 encoder → S3-DiT denoise loop with CFG → VAE decode) via
/// <see cref="RustImageGen"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase
{
    private RustImageGen? _engine;

    [ObservableProperty] private string _modelDir = string.Empty;
    [ObservableProperty] private string _prompt = "a watercolor llama wearing a tiny gear-shaped hat";
    [ObservableProperty] private string _negativePrompt = string.Empty;
    [ObservableProperty] private double _cfgScale = 4.0;
    [ObservableProperty] private double _seed = 42;
    [ObservableProperty] private double _steps = 8;
    [ObservableProperty] private double _latentSize = 64; // image px = latent × 8 → 512²

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    private bool _isBusy;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    private bool _isModelLoaded;

    [ObservableProperty] private string _status =
        "Image generation (Z-Image-Turbo). Load a model directory to begin.";

    [ObservableProperty] private double _progress;
    [ObservableProperty] private Bitmap? _generatedImage;

    private bool CanLoad => !IsBusy;

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadAsync()
    {
        if (string.IsNullOrWhiteSpace(ModelDir)) { Status = "Pick an image-model directory first."; return; }
        IsBusy = true;
        Status = "Loading image model… (streaming weights)";
        try
        {
            _engine ??= new RustImageGen();
            string dir = ModelDir.Trim();
            await Task.Run(() => _engine.LoadBlobs(dir));
            IsModelLoaded = true;
            Status = "Image model loaded.";
        }
        catch (Exception e)
        {
            Status = "Load failed: " + e.Message;
        }
        finally { IsBusy = false; }
    }

    private bool CanGenerate => IsModelLoaded && !IsBusy && !string.IsNullOrWhiteSpace(Prompt);

    [RelayCommand(CanExecute = nameof(CanGenerate))]
    private async Task GenerateAsync()
    {
        if (_engine is null) { Status = "Load a model first."; return; }
        IsBusy = true;
        Progress = 0;
        Status = "Generating…";
        try
        {
            string prompt = Prompt.Trim();
            string neg = NegativePrompt.Trim();
            float cfg = (float)CfgScale;
            uint latent = (uint)LatentSize;
            uint steps = (uint)Steps;
            ulong seed = (ulong)Seed;
            void OnProgress(uint step, uint total, string stage)
            {
                double p = total > 0 ? (double)step / total : 0;
                Dispatcher.UIThread.Post(() => { Progress = p; Status = $"{stage} {step}/{total}"; });
            }
            GeneratedImageData img = await Task.Run(() =>
                _engine.Generate(prompt, neg, cfg, latent, steps, seed, OnProgress));
            GeneratedImage = ToBitmap(img);
            Status = $"Done · {img.Width}×{img.Height}";
        }
        catch (Exception e)
        {
            Status = "Generation failed: " + e.Message;
        }
        finally { IsBusy = false; }
    }

    /// <summary>Build a bitmap from tightly-packed RGBA8 pixels.</summary>
    private static WriteableBitmap ToBitmap(GeneratedImageData img)
    {
        var bmp = new WriteableBitmap(
            new PixelSize(img.Width, img.Height), new Vector(96, 96),
            PixelFormat.Rgba8888, AlphaFormat.Unpremul);
        using ILockedFramebuffer fb = bmp.Lock();
        int rowBytes = img.Width * 4;
        for (int y = 0; y < img.Height; y++)
            Marshal.Copy(img.Rgba, y * rowBytes, fb.Address + y * fb.RowBytes, rowBytes);
        return bmp;
    }
}

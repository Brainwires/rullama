using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Rullama.ViewModels;

namespace Rullama.Views;

public partial class ChatView : UserControl
{
    private ScrollViewer? _scroll;

    public ChatView()
    {
        InitializeComponent();
        _scroll = this.FindControl<ScrollViewer>("MessagesScroll");
        if (_scroll is not null)
        {
            // Stick to the bottom while a reply is streaming.
            _scroll.LayoutUpdated += (_, _) =>
            {
                if (DataContext is ChatViewModel { IsBusy: true } && _scroll is not null)
                    _scroll.Offset = new Vector(_scroll.Offset.X, _scroll.Extent.Height);
            };
        }
    }

    private async void Browse_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not ChatViewModel vm)
            return;

        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select a GGUF model (or Ollama blob)",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("GGUF / Ollama blob") { Patterns = new[] { "*.gguf", "sha256-*" } },
                new FilePickerFileType("All files") { Patterns = new[] { "*" } },
            },
        });

        if (files.Count > 0 && files[0].TryGetLocalPath() is { } path)
            vm.ModelPath = path;
    }

    private async void AttachImage_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not ChatViewModel vm)
            return;

        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Attach an image",
            AllowMultiple = false,
            FileTypeFilter = new[] { FilePickerFileTypes.ImageAll },
        });
        if (files.Count == 0)
            return;

        await using System.IO.Stream stream = await files[0].OpenReadAsync();
        using var ms = new System.IO.MemoryStream();
        await stream.CopyToAsync(ms);
        await vm.AttachImageAsync(ms.ToArray());
    }

    private async void LoadAdapter_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not ChatViewModel vm)
            return;
        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Load a LoRA adapter",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Adapter") { Patterns = new[] { "*.safetensors" } } },
        });
        if (files.Count > 0 && files[0].TryGetLocalPath() is { } path)
            await vm.LoadAdapterAsync(path);
    }

    private async void AttachAudio_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not ChatViewModel vm)
            return;

        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Attach audio (WAV)",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("WAV audio") { Patterns = new[] { "*.wav" } } },
        });
        if (files.Count == 0)
            return;

        await using System.IO.Stream stream = await files[0].OpenReadAsync();
        using var ms = new System.IO.MemoryStream();
        await stream.CopyToAsync(ms);
        await vm.AttachAudioAsync(ms.ToArray());
    }

    // Enter sends; Shift+Enter inserts a newline (AcceptsReturn handles the latter).
    private void Composer_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && e.KeyModifiers == KeyModifiers.None)
        {
            e.Handled = true;
            if (DataContext is ChatViewModel vm && vm.SendCommand.CanExecute(null))
                vm.SendCommand.Execute(null);
        }
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

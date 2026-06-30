using System.Collections.Generic;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Rullama.ViewModels;

namespace Rullama.Views;

public partial class VoiceView : UserControl
{
    public VoiceView()
    {
        InitializeComponent();
    }

    private async void UploadRef_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not VoiceViewModel vm)
            return;
        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Reference clip (WAV, 24 kHz mono)",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("WAV") { Patterns = new[] { "*.wav" } } },
        });
        if (files.Count == 0)
            return;
        await using System.IO.Stream s = await files[0].OpenReadAsync();
        using var ms = new System.IO.MemoryStream();
        await s.CopyToAsync(ms);
        vm.SetReference(ms.ToArray(), files[0].Name);
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

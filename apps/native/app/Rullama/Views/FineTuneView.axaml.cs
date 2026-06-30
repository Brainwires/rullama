using System.Collections.Generic;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Rullama.ViewModels;

namespace Rullama.Views;

public partial class FineTuneView : UserControl
{
    public FineTuneView()
    {
        InitializeComponent();
    }

    private async void Browse_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not FineTuneViewModel vm)
            return;
        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select a base GGUF model",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf", "sha256-*" } } },
        });
        if (files.Count > 0 && files[0].TryGetLocalPath() is { } path)
            vm.ModelPath = path;
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

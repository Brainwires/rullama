using System.Collections.Generic;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Rullama.ViewModels;

namespace Rullama.Views;

public partial class ImageView : UserControl
{
    public ImageView() => InitializeComponent();

    private async void Browse_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not ImageViewModel vm) return;

        IReadOnlyList<IStorageFolder> dirs = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select an image-model directory (Ollama MLX blobs)",
            AllowMultiple = false,
        });
        if (dirs.Count > 0 && dirs[0].TryGetLocalPath() is { } path)
            vm.ModelDir = path;
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

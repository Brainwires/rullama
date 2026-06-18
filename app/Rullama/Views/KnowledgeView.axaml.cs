using System.Collections.Generic;
using System.IO;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Rullama.ViewModels;

namespace Rullama.Views;

public partial class KnowledgeView : UserControl
{
    public KnowledgeView()
    {
        InitializeComponent();
    }

    private async void Upload_Click(object? sender, RoutedEventArgs e)
    {
        TopLevel? top = TopLevel.GetTopLevel(this);
        if (top is null || DataContext is not KnowledgeViewModel vm)
            return;

        IReadOnlyList<IStorageFile> files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Index a text file",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Text") { Patterns = new[] { "*.txt", "*.md" } } },
        });
        if (files.Count == 0)
            return;

        using StreamReader reader = new(await files[0].OpenReadAsync());
        string text = await reader.ReadToEndAsync();
        await vm.IndexTextAsync(files[0].Name, text);
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

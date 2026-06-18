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
            Title = "Index a document",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Text / PDF") { Patterns = new[] { "*.txt", "*.md", "*.pdf" } } },
        });
        if (files.Count == 0)
            return;

        IStorageFile file = files[0];
        string text;
        if (file.Name.EndsWith(".pdf", System.StringComparison.OrdinalIgnoreCase))
        {
            await using System.IO.Stream s = await file.OpenReadAsync();
            using var ms = new System.IO.MemoryStream();
            await s.CopyToAsync(ms);
            text = Rullama.Services.PdfText.Extract(ms.ToArray());
        }
        else
        {
            using StreamReader reader = new(await file.OpenReadAsync());
            text = await reader.ReadToEndAsync();
        }
        await vm.IndexTextAsync(file.Name, text);
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

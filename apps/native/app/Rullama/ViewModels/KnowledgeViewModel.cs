using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>Knowledge base: index documents, semantic search (M7).</summary>
public partial class KnowledgeViewModel : ViewModelBase
{
    private readonly RagService _rag;

    public ObservableCollection<DocumentRow> Documents { get; } = new();
    public ObservableCollection<SearchHit> Results { get; } = new();

    [ObservableProperty] private string _pasteText = string.Empty;
    [ObservableProperty] private string _query = string.Empty;
    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private double _progress;
    [ObservableProperty] private string _status;

    public bool Available => RagService.Available;

    public KnowledgeViewModel() : this(new RagService()) { }

    public KnowledgeViewModel(RagService rag)
    {
        _rag = rag;
        _status = Available
            ? "Ready."
            : "Embedding model not found — place embeddinggemma-300m.gguf in the app data dir.";
        RefreshDocuments();
    }

    public void RefreshDocuments()
    {
        Documents.Clear();
        if (Available)
            foreach (DocumentRow d in _rag.Documents()) Documents.Add(d);
    }

    /// <summary>Index arbitrary text as a named document (also used by file upload).</summary>
    public async Task IndexTextAsync(string name, string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        IsBusy = true;
        Progress = 0;
        Status = $"Indexing {name}…";
        try
        {
            var p = new Progress<double>(v => Progress = v);
            int n = await _rag.IndexTextAsync(name, text, p, CancellationToken.None);
            Status = $"Indexed {name} ({n} chunks).";
            RefreshDocuments();
        }
        catch (Exception e) { Status = "Index failed: " + e.Message; }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private async Task IndexPasteAsync()
    {
        string text = PasteText.Trim();
        if (text.Length == 0) return;
        string firstLine = text.Split('\n')[0].Trim();
        string name = firstLine.Length > 40 ? firstLine[..40] + "…" : firstLine.Length > 0 ? firstLine : "Pasted text";
        PasteText = string.Empty;
        await IndexTextAsync(name, text);
    }

    [RelayCommand]
    private async Task SearchAsync()
    {
        string q = Query.Trim();
        if (q.Length == 0 || !Available) return;
        IsBusy = true;
        Status = "Searching…";
        try
        {
            var hits = await _rag.SearchAsync(q, 8, CancellationToken.None);
            Results.Clear();
            foreach (SearchHit h in hits) Results.Add(h);
            Status = $"{Results.Count} result(s).";
        }
        catch (Exception e) { Status = "Search failed: " + e.Message; }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void DeleteDocument(DocumentRow doc)
    {
        _rag.DeleteDocument(doc.Id);
        Documents.Remove(doc);
        Results.Clear();
    }
}

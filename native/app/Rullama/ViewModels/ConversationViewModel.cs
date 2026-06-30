using System;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Rullama.ViewModels;

/// <summary>A conversation row in the history sidebar.</summary>
public partial class ConversationViewModel : ViewModelBase
{
    public string Id { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(RelativeTime))]
    private long _updatedAt;

    [ObservableProperty]
    private string _title;

    [ObservableProperty]
    private bool _isActive;

    /// <summary>This conversation has a generation running (M11).</summary>
    [ObservableProperty]
    private bool _isRunning;

    /// <summary>This conversation has a queued (not-yet-running) generation (M11).</summary>
    [ObservableProperty]
    private bool _isQueued;

    public ConversationViewModel(string id, string title, long updatedAt)
    {
        Id = id;
        _title = title;
        _updatedAt = updatedAt;
    }

    public string RelativeTime => Relative(UpdatedAt);

    private static string Relative(long unixMs)
    {
        var then = DateTimeOffset.FromUnixTimeMilliseconds(unixMs);
        TimeSpan d = DateTimeOffset.UtcNow - then;
        if (d.TotalSeconds < 60) return "just now";
        if (d.TotalMinutes < 60) return $"{(int)d.TotalMinutes}m ago";
        if (d.TotalHours < 24) return $"{(int)d.TotalHours}h ago";
        if (d.TotalDays < 7) return $"{(int)d.TotalDays}d ago";
        return then.LocalDateTime.ToString("yyyy-MM-dd");
    }
}

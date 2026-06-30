using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using Rullama.Services;

namespace Rullama.ViewModels;

/// <summary>One chat turn. <see cref="Content"/> grows during streaming;
/// <see cref="DisplayContent"/> is the markdown shown (tool markers cleaned).</summary>
public partial class ChatMessageViewModel : ViewModelBase
{
    public string Role { get; }
    public bool IsUser => Role == "user";
    public string RoleLabel => Role.ToUpperInvariant();

    public Bitmap? Image { get; init; }
    public bool HasImage => Image is not null;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayContent))]
    private string _content;

    /// <summary>Raw content with tool_call/tool_response markers rendered for display.</summary>
    public string DisplayContent => IsUser ? Content : ToolCalling.CleanForDisplay(Content);

    public ChatMessageViewModel(string role, string content)
    {
        Role = role;
        _content = content;
    }
}

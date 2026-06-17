using CommunityToolkit.Mvvm.ComponentModel;

namespace Rullama.ViewModels;

/// <summary>One chat turn. <see cref="Content"/> grows during streaming.</summary>
public partial class ChatMessageViewModel : ViewModelBase
{
    public string Role { get; }
    public bool IsUser => Role == "user";
    public string RoleLabel => Role.ToUpperInvariant();

    [ObservableProperty]
    private string _content;

    public ChatMessageViewModel(string role, string content)
    {
        Role = role;
        _content = content;
    }
}

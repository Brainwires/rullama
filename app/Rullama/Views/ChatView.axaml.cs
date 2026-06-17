using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
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

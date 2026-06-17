using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace Rullama.Views;

public partial class VoiceView : UserControl
{
    public VoiceView()
    {
        InitializeComponent();
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}

using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Markup.Xaml;

namespace Rullama.Controls;

/// <summary>Labelled slider matching the PWA: uppercase label + mono value + track.</summary>
public partial class LabeledSlider : UserControl
{
    public static readonly StyledProperty<string> LabelProperty =
        AvaloniaProperty.Register<LabeledSlider, string>(nameof(Label), string.Empty);

    public static readonly StyledProperty<double> ValueProperty =
        AvaloniaProperty.Register<LabeledSlider, double>(
            nameof(Value), defaultBindingMode: BindingMode.TwoWay);

    public static readonly StyledProperty<double> MinimumProperty =
        AvaloniaProperty.Register<LabeledSlider, double>(nameof(Minimum), 0d);

    public static readonly StyledProperty<double> MaximumProperty =
        AvaloniaProperty.Register<LabeledSlider, double>(nameof(Maximum), 1d);

    public static readonly StyledProperty<double> StepProperty =
        AvaloniaProperty.Register<LabeledSlider, double>(nameof(Step), 0.01d);

    public static readonly StyledProperty<string> FormatProperty =
        AvaloniaProperty.Register<LabeledSlider, string>(nameof(Format), "0.00");

    private string _valueText = string.Empty;
    public static readonly DirectProperty<LabeledSlider, string> ValueTextProperty =
        AvaloniaProperty.RegisterDirect<LabeledSlider, string>(nameof(ValueText), o => o._valueText);

    public string Label { get => GetValue(LabelProperty); set => SetValue(LabelProperty, value); }
    public double Value { get => GetValue(ValueProperty); set => SetValue(ValueProperty, value); }
    public double Minimum { get => GetValue(MinimumProperty); set => SetValue(MinimumProperty, value); }
    public double Maximum { get => GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }
    public double Step { get => GetValue(StepProperty); set => SetValue(StepProperty, value); }
    public string Format { get => GetValue(FormatProperty); set => SetValue(FormatProperty, value); }

    public string ValueText
    {
        get => _valueText;
        private set => SetAndRaise(ValueTextProperty, ref _valueText, value);
    }

    public LabeledSlider()
    {
        AvaloniaXamlLoader.Load(this);
        UpdateText();
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.Property == ValueProperty || e.Property == FormatProperty)
            UpdateText();
    }

    private void UpdateText() => ValueText = Value.ToString(Format, CultureInfo.InvariantCulture);
}

using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Rullama.Controls;

/// <summary>A tiny dependency-free line chart for a sequence of values (e.g. training loss).</summary>
public sealed class Sparkline : Control
{
    public static readonly StyledProperty<System.Collections.IEnumerable?> ValuesProperty =
        AvaloniaProperty.Register<Sparkline, System.Collections.IEnumerable?>(nameof(Values));

    public static readonly StyledProperty<IBrush> StrokeProperty =
        AvaloniaProperty.Register<Sparkline, IBrush>(nameof(Stroke), Brushes.DodgerBlue);

    public System.Collections.IEnumerable? Values
    {
        get => GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }

    public IBrush Stroke
    {
        get => GetValue(StrokeProperty);
        set => SetValue(StrokeProperty, value);
    }

    private INotifyCollectionChanged? _subscribed;

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.Property == ValuesProperty)
        {
            if (_subscribed is not null) _subscribed.CollectionChanged -= OnCollectionChanged;
            _subscribed = Values as INotifyCollectionChanged;
            if (_subscribed is not null) _subscribed.CollectionChanged += OnCollectionChanged;
            InvalidateVisual();
        }
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) => InvalidateVisual();

    public override void Render(DrawingContext ctx)
    {
        base.Render(ctx);
        if (Values is null) return;
        List<double> vals = Values.Cast<object>().Select(System.Convert.ToDouble).ToList();
        if (vals.Count < 2) return;

        double min = vals.Min(), max = vals.Max();
        double range = max - min < 1e-9 ? 1 : max - min;
        double w = Bounds.Width, h = Bounds.Height;
        if (w <= 0 || h <= 0) return;

        var geo = new StreamGeometry();
        using (StreamGeometryContext c = geo.Open())
        {
            for (int i = 0; i < vals.Count; i++)
            {
                double x = w * i / (vals.Count - 1);
                double y = h - (vals[i] - min) / range * h;
                var p = new Point(x, y);
                if (i == 0) c.BeginFigure(p, false);
                else c.LineTo(p);
            }
            c.EndFigure(false);
        }
        ctx.DrawGeometry(null, new Pen(Stroke, 1.5), geo);
    }
}

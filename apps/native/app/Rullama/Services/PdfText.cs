using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace Rullama.Services;

/// <summary>Extracts plain text from a PDF (PdfPig), page by page.</summary>
public static class PdfText
{
    public static string Extract(byte[] pdfBytes)
    {
        var sb = new StringBuilder();
        using PdfDocument doc = PdfDocument.Open(pdfBytes);
        foreach (Page page in doc.GetPages())
            sb.AppendLine(page.Text).AppendLine();
        return sb.ToString();
    }
}

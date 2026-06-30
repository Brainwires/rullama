using System.Collections.Generic;
using System.Text;

namespace Rullama.Services;

/// <summary>Simple paragraph-aware chunker (~target chars, with overlap).</summary>
public static class TextSplit
{
    public static List<string> Chunk(string text, int targetChars = 800, int overlap = 120)
    {
        var chunks = new List<string>();
        string[] paras = text.Replace("\r\n", "\n").Split("\n\n");
        var cur = new StringBuilder();

        void Flush()
        {
            string s = cur.ToString().Trim();
            if (s.Length > 0) chunks.Add(s);
            cur.Clear();
        }

        foreach (string para in paras)
        {
            string p = para.Trim();
            if (p.Length == 0) continue;

            if (p.Length > targetChars)
            {
                Flush();
                for (int i = 0; i < p.Length; i += targetChars - overlap)
                {
                    int len = System.Math.Min(targetChars, p.Length - i);
                    chunks.Add(p.Substring(i, len).Trim());
                    if (i + len >= p.Length) break;
                }
                continue;
            }

            if (cur.Length + p.Length + 2 > targetChars) Flush();
            if (cur.Length > 0) cur.Append("\n\n");
            cur.Append(p);
        }
        Flush();
        return chunks;
    }
}

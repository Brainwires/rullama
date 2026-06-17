// PDF → per-page text extraction via pdf.js. Text-layer only (no OCR);
// scanned PDFs without a text layer yield empty pages.
//
// pdf.js needs a worker; we point it at the bundled worker entry so Vite
// serves it. The import is dynamic so the ~1 MB pdf.js bundle only loads
// when the user actually drops a PDF.

export interface PdfPage {
    text: string;
    page: number;
}

export async function extractPdfText(file: File | ArrayBuffer): Promise<PdfPage[]> {
    const pdfjs = await import("pdfjs-dist");
    // Vite resolves the worker URL; pdf.js v4+ uses a module worker.
    const workerUrl = (await import("pdfjs-dist/build/pdf.worker.mjs?url")).default;
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

    const data = file instanceof ArrayBuffer ? file : await file.arrayBuffer();
    const doc = await pdfjs.getDocument({ data: new Uint8Array(data) }).promise;
    const pages: PdfPage[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
        const page = await doc.getPage(i);
        const content = await page.getTextContent();
        const text = content.items
            .map((it) => ("str" in it ? (it as { str: string }).str : ""))
            .join(" ")
            .replace(/\s+/g, " ")
            .trim();
        pages.push({ text, page: i });
    }
    try { await (doc as unknown as { cleanup: () => Promise<void> }).cleanup(); } catch { /* */ }
    return pages.filter((p) => p.text.length > 0);
}

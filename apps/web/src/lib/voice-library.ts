/// Persistent library of cloned voices (StyleTTS2 256-d style vectors). Each vector is
/// tiny (~1 KB) so localStorage is plenty; surfaced in the Voice tab + Fine-tune panel.

export interface SavedVoice {
    id: string;
    name: string;
    vec: number[]; // 256 floats
    createdAt: number;
}

const KEY = "rullama:cloned-voices";
const EVENT = "rullama:voices-changed";

export function listVoices(): SavedVoice[] {
    try {
        const v = JSON.parse(localStorage.getItem(KEY) ?? "[]");
        return Array.isArray(v) ? v : [];
    } catch {
        return [];
    }
}

function persist(voices: SavedVoice[]): void {
    localStorage.setItem(KEY, JSON.stringify(voices));
    window.dispatchEvent(new Event(EVENT));
}

export function addVoice(name: string, vec: Float32Array): SavedVoice {
    const sv: SavedVoice = { id: crypto.randomUUID(), name: name.trim() || "My voice", vec: Array.from(vec), createdAt: Date.now() };
    persist([...listVoices(), sv]);
    return sv;
}

export function removeVoice(id: string): void {
    persist(listVoices().filter((v) => v.id !== id));
}

export function renameVoice(id: string, name: string): void {
    persist(listVoices().map((v) => (v.id === id ? { ...v, name: name.trim() || v.name } : v)));
}

export function voiceVec(v: SavedVoice): Float32Array {
    return new Float32Array(v.vec);
}

/** Subscribe to library changes (same-tab custom event + cross-tab storage event). */
export function onVoicesChanged(cb: () => void): () => void {
    const h = () => cb();
    window.addEventListener(EVENT, h);
    window.addEventListener("storage", h);
    return () => {
        window.removeEventListener(EVENT, h);
        window.removeEventListener("storage", h);
    };
}

/** Read a 256-float .f32 export (1024 bytes) into a vector. */
export async function importVoiceFile(f: File): Promise<Float32Array> {
    const buf = await f.arrayBuffer();
    const vec = new Float32Array(buf);
    if (vec.length !== 256) throw new Error(`expected a 256-float .f32 voice, got ${vec.length} floats`);
    return vec;
}

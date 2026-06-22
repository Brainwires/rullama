// Cloud API key vault — encrypts BYOK keys at rest on the device.
//
// The key only ever leaves the device as the per-request `X-Cloud-Key` header
// the proxy forwards (and never persists). On the DEVICE we don't want it
// sitting in plaintext `localStorage` (unlike the low-stakes weather/news
// keys), so we store the ciphertext in OPFS, encrypted with a
// **non-extractable** AES-GCM `CryptoKey` held in IndexedDB. Non-extractable
// means JS can call `decrypt` with the key but can never read its raw bytes —
// so the plaintext API key is not recoverable from disk, a backup, or a casual
// devtools poke.
//
// Honest scope: this protects AT REST. It can't protect against an XSS'd page
// at use-time (which can call decrypt) or the proxy seeing the key in flight
// (it must, to set Authorization). rullama ships no third-party scripts, which
// keeps the XSS surface small.

import type { CloudProvider } from "./types";

const IDB_NAME = "rullama-cloud-vault";
const IDB_STORE = "keys";
const VAULT_KEY_ID = "aes-gcm-v1";
const OPFS_DIR = "rullama-cloud-keys";

function idbOpen(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(IDB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function idbGet<T>(db: IDBDatabase, key: string): Promise<T | undefined> {
    return new Promise((resolve, reject) => {
        const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(key);
        req.onsuccess = () => resolve(req.result as T | undefined);
        req.onerror = () => reject(req.error);
    });
}

function idbPut(db: IDBDatabase, key: string, value: unknown): Promise<void> {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, "readwrite");
        tx.objectStore(IDB_STORE).put(value, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

/** The non-extractable AES-GCM key. Created once, then reused. Stored as a
 *  CryptoKey object (structured-clonable) so its raw bytes never surface. */
async function getOrCreateVaultKey(): Promise<CryptoKey> {
    const db = await idbOpen();
    try {
        const existing = await idbGet<CryptoKey>(db, VAULT_KEY_ID);
        if (existing) return existing;
        const key = await crypto.subtle.generateKey(
            { name: "AES-GCM", length: 256 },
            false, // non-extractable — the whole point
            ["encrypt", "decrypt"],
        );
        await idbPut(db, VAULT_KEY_ID, key);
        return key;
    } finally {
        db.close();
    }
}

async function vaultDir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(OPFS_DIR, { create: true });
}

const fileName = (provider: CloudProvider) => `${provider}.bin`;

/** Event name fired whenever a provider key is saved/cleared, so UI that gates
 *  on "is a key set?" (e.g. the model picker's Load button) can re-check. */
export const CLOUD_KEY_CHANGE_EVENT = "rullama:cloudkeychange";

function notifyKeyChange(provider: CloudProvider): void {
    try { window.dispatchEvent(new CustomEvent(CLOUD_KEY_CHANGE_EVENT, { detail: { provider } })); }
    catch { /* non-browser / no window */ }
}

/** Encrypt + store the key for a provider. An empty string clears it. */
export async function putCloudKey(provider: CloudProvider, plaintext: string): Promise<void> {
    const dir = await vaultDir();
    const name = fileName(provider);
    if (!plaintext) {
        try { await dir.removeEntry(name); } catch { /* already absent */ }
        notifyKeyChange(provider);
        return;
    }
    const cryptoKey = await getOrCreateVaultKey();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = new Uint8Array(
        await crypto.subtle.encrypt(
            { name: "AES-GCM", iv },
            cryptoKey,
            new TextEncoder().encode(plaintext),
        ),
    );
    // On-disk layout: [12-byte IV][ciphertext].
    const out = new Uint8Array(iv.length + ct.length);
    out.set(iv, 0);
    out.set(ct, iv.length);
    const fh = await dir.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.truncate(0);
    await w.write(out);
    await w.close();
    notifyKeyChange(provider);
}

/** Read + decrypt the key for a provider. Returns "" when absent or
 *  undecryptable (e.g. the vault key was cleared) — callers treat "" as
 *  "no key set". */
export async function getCloudKey(provider: CloudProvider): Promise<string> {
    try {
        const dir = await vaultDir();
        const fh = await dir.getFileHandle(fileName(provider), { create: false });
        const buf = new Uint8Array(await (await fh.getFile()).arrayBuffer());
        if (buf.length < 13) return "";
        const iv = buf.slice(0, 12);
        const ct = buf.slice(12);
        const cryptoKey = await getOrCreateVaultKey();
        const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, cryptoKey, ct);
        return new TextDecoder().decode(pt);
    } catch {
        return "";
    }
}

/** Whether a key file exists for the provider — for the "key set?" UI hint,
 *  without decrypting. */
export async function hasCloudKey(provider: CloudProvider): Promise<boolean> {
    try {
        const dir = await vaultDir();
        await dir.getFileHandle(fileName(provider), { create: false });
        return true;
    } catch {
        return false;
    }
}

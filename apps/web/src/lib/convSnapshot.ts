// RLCV — per-conversation KV-snapshot envelope codec.
//
// A persisted conversation snapshot bundles the resident TOKEN sequence
// alongside the model's KV/sampler bytes. The token list is what lets a
// reopened conversation skip the prefill: on restore the core seeds its
// `residentIds` tracker from it, so the next turn's kvReusePlan matches
// the rendered prefix and only feeds the new suffix.
//
// Layout (little-endian):
//   [0..4]   "RLCV" magic
//   [4]      version = 1
//   [5..8]   reserved
//   [8..12]  idsLen (u32)        — number of resident tokens
//   [12..]   ids (u32 × idsLen)
//   [..]     kvBytes             — opaque RLMS blob (sampler + RLKV)
//
// Pure functions (no wasm / OPFS) so they unit-test in a node env.

export const RLCV_VERSION = 1;
const HEADER_BYTES = 12;

/** Encode resident token ids + the model's KV-state blob into one buffer. */
export function encodeRlcvEnvelope(
    ids: ArrayLike<number>,
    kvBytes: Uint8Array,
): Uint8Array {
    const idsArr = ids instanceof Uint32Array ? ids : Uint32Array.from(ids);
    const idsBytes = new Uint8Array(idsArr.buffer, idsArr.byteOffset, idsArr.byteLength);
    const out = new Uint8Array(HEADER_BYTES + idsBytes.length + kvBytes.length);
    const dv = new DataView(out.buffer);
    // "RLCV" magic, byte-for-byte (avoids endian ambiguity).
    out[0] = 0x52; out[1] = 0x4c; out[2] = 0x43; out[3] = 0x56;
    dv.setUint8(4, RLCV_VERSION);
    dv.setUint32(8, idsArr.length, true);
    out.set(idsBytes, HEADER_BYTES);
    out.set(kvBytes, HEADER_BYTES + idsBytes.length);
    return out;
}

/** Inverse of {@link encodeRlcvEnvelope}. Throws on bad magic / version /
 *  truncation. `ids` is copied out (alignment-safe); `kvBytes` is a view. */
export function decodeRlcvEnvelope(bytes: Uint8Array): { ids: Uint32Array; kvBytes: Uint8Array } {
    if (bytes.length < HEADER_BYTES
        || bytes[0] !== 0x52 || bytes[1] !== 0x4c || bytes[2] !== 0x43 || bytes[3] !== 0x56) {
        throw new Error("RLCV: bad magic");
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = dv.getUint8(4);
    if (version !== RLCV_VERSION) throw new Error(`RLCV: unknown version ${version}`);
    const idsLen = dv.getUint32(8, true);
    const idsByteLen = idsLen * 4;
    if (bytes.length < HEADER_BYTES + idsByteLen) throw new Error("RLCV: truncated ids");
    // Copy (slice) so the Uint32Array doesn't rely on the source being
    // 4-byte aligned within its underlying ArrayBuffer.
    const ids = new Uint32Array(bytes.slice(HEADER_BYTES, HEADER_BYTES + idsByteLen).buffer);
    const kvBytes = bytes.subarray(HEADER_BYTES + idsByteLen);
    return { ids, kvBytes };
}

/** Peek the resident-token count from an envelope header without copying
 *  the payload (used for the sidecar's tokenCount). */
export function rlcvTokenCount(bytes: Uint8Array): number {
    return new DataView(bytes.buffer, bytes.byteOffset, HEADER_BYTES).getUint32(8, true);
}

import { describe, it, expect } from "vitest";
import { encodeRlcvEnvelope, decodeRlcvEnvelope, rlcvTokenCount, RLCV_VERSION } from "./convSnapshot";

describe("RLCV conversation-snapshot envelope", () => {
    it("round-trips ids and kv payload", () => {
        const ids = [1, 2, 3, 105, 2364, 107, 65535, 4294967295]; // incl. max u32
        const kv = new Uint8Array([10, 20, 30, 40, 50]);
        const env = encodeRlcvEnvelope(ids, kv);
        const out = decodeRlcvEnvelope(env);
        expect(Array.from(out.ids)).toEqual(ids);
        expect(Array.from(out.kvBytes)).toEqual(Array.from(kv));
    });

    it("round-trips an empty kv payload and a single token", () => {
        const env = encodeRlcvEnvelope([42], new Uint8Array(0));
        const out = decodeRlcvEnvelope(env);
        expect(Array.from(out.ids)).toEqual([42]);
        expect(out.kvBytes.length).toBe(0);
    });

    it("accepts a Uint32Array as ids input", () => {
        const ids = Uint32Array.from([7, 8, 9]);
        const env = encodeRlcvEnvelope(ids, new Uint8Array([1]));
        expect(Array.from(decodeRlcvEnvelope(env).ids)).toEqual([7, 8, 9]);
    });

    it("writes the version byte", () => {
        const env = encodeRlcvEnvelope([1], new Uint8Array(0));
        expect(env[4]).toBe(RLCV_VERSION);
    });

    it("peeks tokenCount without full decode", () => {
        const env = encodeRlcvEnvelope([1, 2, 3, 4, 5], new Uint8Array([9, 9]));
        expect(rlcvTokenCount(env)).toBe(5);
    });

    it("decodes correctly when the envelope sits at a non-zero byteOffset", () => {
        const ids = [11, 22, 33];
        const env = encodeRlcvEnvelope(ids, new Uint8Array([1, 2, 3]));
        // Embed the envelope inside a larger buffer at an odd offset so the
        // u32 ids view can't alias the source buffer.
        const backing = new Uint8Array(env.length + 3);
        backing.set(env, 3);
        const view = backing.subarray(3);
        const out = decodeRlcvEnvelope(view);
        expect(Array.from(out.ids)).toEqual(ids);
        expect(Array.from(out.kvBytes)).toEqual([1, 2, 3]);
    });

    it("rejects bad magic", () => {
        const env = encodeRlcvEnvelope([1], new Uint8Array(0));
        env[0] = 0x00;
        expect(() => decodeRlcvEnvelope(env)).toThrow(/magic/);
    });

    it("rejects an unknown version", () => {
        const env = encodeRlcvEnvelope([1], new Uint8Array(0));
        env[4] = 99;
        expect(() => decodeRlcvEnvelope(env)).toThrow(/version/);
    });

    it("rejects a truncated ids section", () => {
        const env = encodeRlcvEnvelope([1, 2, 3, 4], new Uint8Array(0));
        // Drop bytes so the declared idsLen (4) overruns the buffer.
        expect(() => decodeRlcvEnvelope(env.subarray(0, 14))).toThrow(/truncated/);
    });
});

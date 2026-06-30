// AudioWorkletProcessor that buffers samples into 20 ms frames at the
// AudioContext's sample rate (rullama opens at 16 kHz → 320 samples/frame),
// computes RMS in dBFS per frame, and posts each frame to the main thread.
//
// The processor itself does NOT decide when to stop recording; it just
// streams frame-grained energy + audio. The main-thread hook
// (useMicCapture) runs the silence-detection state machine. Keeping the
// worklet stateless makes the math (RMS threshold, silence cutoff,
// pre-roll size) trivial to tune from React without rebuilding the
// worklet file.
//
// Frame size choice: 20 ms is the same window the rullama-framework
// EnergyVad uses and is short enough to detect end-of-utterance within
// the 800 ms silence budget while long enough that one syllable's RMS
// is a stable signal.

class VadWorklet extends AudioWorkletProcessor {
    constructor() {
        super();
        // sampleRate is a global injected into AudioWorkletGlobalScope.
        // At 16 kHz this is 320; at 48 kHz it's 960.
        this.frameSize = Math.max(1, Math.round(sampleRate * 0.02));
        this.buf = new Float32Array(this.frameSize);
        this.idx = 0;
    }

    process(inputs) {
        // inputs[0] is the input port; channel 0 is mono.
        const ch0 = inputs[0]?.[0];
        if (!ch0) return true;

        for (let i = 0; i < ch0.length; i++) {
            this.buf[this.idx++] = ch0[i];
            if (this.idx >= this.frameSize) {
                let sumSq = 0;
                for (let j = 0; j < this.frameSize; j++) {
                    sumSq += this.buf[j] * this.buf[j];
                }
                const meanSq = sumSq / this.frameSize;
                // +1e-10 to keep log10 finite during dead silence.
                const rmsDb = 10 * Math.log10(meanSq + 1e-10);
                // .slice() because the main thread will hold this past
                // the next process() call. ArrayBufferView transfers
                // would be slightly cheaper but worklet→main port doesn't
                // support transferables in all browsers.
                this.port.postMessage({ rmsDb, samples: this.buf.slice() });
                this.idx = 0;
            }
        }
        return true;
    }
}

registerProcessor('vad-worklet', VadWorklet);

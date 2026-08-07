class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phase = 0;
    this.targetRate = 16000;
    this.samples = [];
    this.levelCounter = 0;
    this.levelEnergy = 0;
    this.levelSamples = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") this.flush();
    };
  }

  flush() {
    if (!this.samples.length) return;
    const output = new Int16Array(this.samples);
    this.samples = [];
    this.port.postMessage({ type: "audio", buffer: output.buffer }, [output.buffer]);
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let energy = 0;
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      energy += sample * sample;
      this.phase += this.targetRate;
      if (this.phase >= sampleRate) {
        this.phase -= sampleRate;
        this.samples.push(sample < 0 ? sample * 32768 : sample * 32767);
        if (this.samples.length >= 320) {
          const output = new Int16Array(this.samples.splice(0, 320));
          this.port.postMessage({ type: "audio", buffer: output.buffer }, [output.buffer]);
        }
      }
    }
    this.levelEnergy += energy;
    this.levelSamples += input.length;
    this.levelCounter += input.length;
    if (this.levelCounter >= sampleRate / 10) {
      this.levelCounter = 0;
      const value = Math.sqrt(this.levelEnergy / Math.max(1, this.levelSamples));
      this.levelEnergy = 0;
      this.levelSamples = 0;
      this.port.postMessage({ type: "level", value });
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);

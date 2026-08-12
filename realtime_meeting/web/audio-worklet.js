class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputRate = sampleRate;
    this.targetRate = 16000;
    this.packetSamples = this.targetRate * 40 / 1000;
    this.ratio = this.inputRate / this.targetRate;
    this.halfTaps = 16;
    this.inputBuffer = [];
    this.sourcePosition = this.halfTaps;
    this.output = [];
    this.levelCounter = 0;
    this.levelEnergy = 0;
    this.levelSamples = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") this.flush();
    };
  }

  _sinc(value) {
    if (Math.abs(value) < 1e-8) return 1;
    const piValue = Math.PI * value;
    return Math.sin(piValue) / piValue;
  }

  _resample() {
    const cutoff = Math.min(0.5, 0.45 / Math.max(1, this.ratio));
    while (this.sourcePosition + this.halfTaps < this.inputBuffer.length) {
      const center = Math.floor(this.sourcePosition);
      let sum = 0;
      let weightSum = 0;
      for (let offset = -this.halfTaps + 1; offset <= this.halfTaps; offset += 1) {
        const index = center + offset;
        if (index < 0 || index >= this.inputBuffer.length) continue;
        const distance = index - this.sourcePosition;
        const window = 0.5 + 0.5 * Math.cos((Math.PI * distance) / this.halfTaps);
        const weight = 2 * cutoff * this._sinc(2 * cutoff * distance) * window;
        sum += this.inputBuffer[index] * weight;
        weightSum += weight;
      }
      const value = weightSum ? sum / weightSum : 0;
      this.output.push(Math.max(-1, Math.min(1, value)));
      this.sourcePosition += this.ratio;
    }
    const remove = Math.max(0, Math.floor(this.sourcePosition) - this.halfTaps);
    if (remove > 0) {
      this.inputBuffer.splice(0, remove);
      this.sourcePosition -= remove;
    }
  }

  _emitFrames() {
    while (this.output.length >= this.packetSamples) {
      const frame = new Int16Array(this.packetSamples);
      for (let index = 0; index < frame.length; index += 1) {
        const sample = this.output[index];
        frame[index] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      this.output.splice(0, this.packetSamples);
      this.port.postMessage({ type: "audio", buffer: frame.buffer }, [frame.buffer]);
    }
  }

  flush() {
    if (this.inputBuffer.length) {
      this.inputBuffer.push(...new Array(this.halfTaps + 2).fill(0));
      this._resample();
    }
    this._emitFrames();
    if (this.output.length) {
      const frame = new Int16Array(this.output.length);
      for (let index = 0; index < frame.length; index += 1) {
        const sample = this.output[index];
        frame[index] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      this.output = [];
      this.port.postMessage({ type: "audio", buffer: frame.buffer }, [frame.buffer]);
    }
    this.inputBuffer = [];
    this.sourcePosition = this.halfTaps;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let energy = 0;
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      this.inputBuffer.push(sample);
      energy += sample * sample;
    }
    this._resample();
    this._emitFrames();

    this.levelEnergy += energy;
    this.levelSamples += input.length;
    this.levelCounter += input.length;
    if (this.levelCounter >= this.inputRate / 10) {
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

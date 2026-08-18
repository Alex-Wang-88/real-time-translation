class MeetingCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetRate || 16000;
    this.packetSamples = opts.packetSamples || 640;
    this.inputRate = sampleRate;
    this.output = [];
    this.filterRadius = 12;
    this.resampleInput = new Array(this.filterRadius).fill(0);
    this.resamplePosition = this.filterRadius;
    this.sequence = 0;
    this.thresholdPercent = Number(opts.thresholdPercent) || 0;
    this.thresholdRms = 0;
    this.setThreshold(this.thresholdPercent);
    this.port.onmessage = (event) => {
      if (event.data?.type === "volume_threshold") this.setThreshold(event.data.percent);
    };
    this.port.postMessage({ type: "ready", inputRate: this.inputRate, targetRate: this.targetRate });
  }

  setThreshold(percent) {
    const value = Math.max(0, Math.min(30, Number(percent) || 0));
    this.thresholdPercent = value;
    this.thresholdRms = value / 100 / 3;
  }

  resampleAt(position) {
    const center = Math.floor(position);
    const cutoff = Math.min(1, this.targetRate / this.inputRate) * 0.94;
    let weighted = 0;
    let weightSum = 0;
    for (let tap = -this.filterRadius + 1; tap <= this.filterRadius; tap += 1) {
      const index = center + tap;
      if (index < 0 || index >= this.resampleInput.length) continue;
      const distance = index - position;
      if (Math.abs(distance) >= this.filterRadius) continue;
      const scaled = Math.PI * distance * cutoff;
      const sinc = Math.abs(scaled) < 1e-8 ? 1 : Math.sin(scaled) / scaled;
      const window = 0.5 + 0.5 * Math.cos(Math.PI * distance / this.filterRadius);
      const weight = cutoff * sinc * window;
      weighted += this.resampleInput[index] * weight;
      weightSum += weight;
    }
    return weightSum ? weighted / weightSum : 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0].length) return true;
    const length = channels[0].length;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < channels.length; channel += 1) {
      const input = channels[channel];
      for (let index = 0; index < length; index += 1) mono[index] += input[index] / channels.length;
    }
    let sum = 0;
    for (const value of mono) sum += value * value;
    // Keep the original per-audio-block level reporting. The main thread
    // coalesces these updates into one animation-frame render.
    this.port.postMessage({ type: "level", value: Math.min(1, Math.sqrt(sum / Math.max(1, mono.length)) * 3) });
    if (this.inputRate === this.targetRate) {
      this.output.push(...mono);
    } else {
      const ratio = this.inputRate / this.targetRate;
      this.resampleInput.push(...mono);
      while (this.resamplePosition + this.filterRadius < this.resampleInput.length) {
        const sample = this.resampleAt(this.resamplePosition);
        this.output.push(Math.max(-1, Math.min(1, sample)));
        this.resamplePosition += ratio;
      }
    }
    const consumed = Math.max(0, Math.floor(this.resamplePosition) - this.filterRadius);
    if (this.inputRate !== this.targetRate && consumed) {
      this.resampleInput.splice(0, consumed);
      this.resamplePosition -= consumed;
    }
    while (this.output.length >= this.packetSamples) {
      const packet = this.output.splice(0, this.packetSamples);
      const pcm = new ArrayBuffer(4 + packet.length * 2);
      const view = new DataView(pcm);
      view.setUint32(0, this.sequence, true);
      this.sequence = (this.sequence + 1) >>> 0;
      for (let index = 0; index < packet.length; index += 1) {
        const value = packet[index] < 0 ? packet[index] * 32768 : packet[index] * 32767;
        view.setInt16(4 + index * 2, value, true);
      }
      this.port.postMessage({ type: "audio", buffer: pcm }, [pcm]);
    }
    return true;
  }
}

registerProcessor("meeting-capture-processor", MeetingCaptureProcessor);

class MeetingCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetRate || 16000;
    this.packetSamples = opts.packetSamples || 640;
    this.inputRate = sampleRate;
    this.output = [];
    this.resampleInput = [];
    this.resamplePosition = 0;
    this.sequence = 0;
    this.port.postMessage({ type: "ready", inputRate: this.inputRate, targetRate: this.targetRate });
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
    this.port.postMessage({ type: "level", value: Math.min(1, Math.sqrt(sum / Math.max(1, mono.length)) * 3) });
    const ratio = this.inputRate / this.targetRate;
    this.resampleInput.push(...mono);
    while (this.resamplePosition + 1 < this.resampleInput.length) {
      const position = this.resamplePosition;
      const left = Math.floor(position);
      const right = left + 1;
      const fraction = position - left;
      const sample = this.resampleInput[left] * (1 - fraction) + this.resampleInput[right] * fraction;
      this.output.push(Math.max(-1, Math.min(1, sample)));
      this.resamplePosition += ratio;
    }
    const consumed = Math.floor(this.resamplePosition);
    if (consumed) {
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

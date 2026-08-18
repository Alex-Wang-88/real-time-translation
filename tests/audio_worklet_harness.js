const fs = require("fs");
const vm = require("vm");

const workletPath = process.argv[2];
const inputRate = Number(process.argv[3]);
const frequency = Number(process.argv[4]);
const amplitude = Number(process.argv[5]);
global.sampleRate = inputRate;
let Processor = null;
global.AudioWorkletProcessor = class {
  constructor() {
    this.port = {
      messages: [],
      onmessage: null,
      postMessage: (message) => this.port.messages.push(message),
    };
  }
};
global.registerProcessor = (_name, implementation) => { Processor = implementation; };
vm.runInThisContext(fs.readFileSync(workletPath, "utf8"), { filename: workletPath });

const processor = new Processor({
  processorOptions: { targetRate: 16000, packetSamples: 160, thresholdPercent: 30 },
});
const total = inputRate;
for (let start = 0; start < total; start += 128) {
  const length = Math.min(128, total - start);
  const block = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    block[index] = amplitude * Math.sin(2 * Math.PI * frequency * (start + index) / inputRate);
  }
  processor.process([[block]]);
}
const samples = [];
for (const message of processor.port.messages) {
  if (message.type !== "audio") continue;
  const view = new DataView(message.buffer);
  for (let offset = 4; offset < view.byteLength; offset += 2) samples.push(view.getInt16(offset, true) / 32768);
}
const settled = samples.slice(Math.min(400, samples.length));
const rms = Math.sqrt(settled.reduce((sum, value) => sum + value * value, 0) / Math.max(1, settled.length));
const levelCount = processor.port.messages.filter((message) => message.type === "level").length;
let maxJump = 0;
for (let index = 1; index < settled.length; index += 1) {
  maxJump = Math.max(maxJump, Math.abs(settled[index] - settled[index - 1]));
}
process.stdout.write(JSON.stringify({ inputRate, frequency, count: samples.length, rms, maxJump, levelCount }));

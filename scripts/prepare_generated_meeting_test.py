"""Prepare a local ASR/translation benchmark from the toolbox meeting output."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path


LANGUAGE_MAP = {
    "普通话": ("zh", "mandarin", "Mandarin"),
    "四川方言": ("zh", "sichuan", "Sichuan"),
    "English": ("en", "unknown", "English"),
    "Deutsch": ("de", "unknown", "German"),
}


TRANSLATIONS = {
    "role_meeting_v3_003": "从客户角度看，订单池为一万三千二百台。六百二十台已经进入下个月的交付窗口，三十七个订单存在可信的延期风险。最敏感的客户正在等待两种使用铜端子的产品给出确定日期。四家德国客户在近期投诉后要求提供可追溯性和明确的纠正措施日期。我已经把确定订单、预测需求以及等待规格确认的订单分开。主要问题是，我们给客户的承诺更新速度已经超过工厂确认产能的速度。",
    "role_meeting_v3_008": "我会优先保障有合同罚则、上市节点承诺以及很大复购可能的客户。第二类客户可以谈分批交付，但必须给出准确数量和日期，不能含糊承诺。如果工厂只能提供理论产能的百分之九十，我会先把产能分给两种铜端子产品以及已经受到投诉影响的客户。剩余订单按照利润、合同风险以及客户是否接受分批交付排序。在发送最终日期之前，我需要生产部门确认一份产能窗口。",
    "role_meeting_v3_015": "产能窗口更新后四十八小时内，我可以反馈所有未解决的交付风险。对于拒绝分批交付的客户，我会在确认日期前说明风险，并要求客户书面决定优先级或额外运费。我还会把客户证据和生产假设放在同一张表里。这样产能或规格一变化就能立即看见，不会在月底变成意外。",
    "role_meeting_v3_019": "明白。客户工作已经在进行中。我已经把确定订单和预测需求分开，并开始联系五个最高风险客户。九月五日前，我会交付一份包含准确日期、分批交付选项、客户证据以及每个高风险客户一项替代方案的承诺表。在客户书面回复或电话纪要记录决定之前，我不会把日期标记为已确认。",
    "role_meeting_v3_004": "从财务和采购角度看，实际制造成本为每台三百一十八元，比预算高十三元。材料价格解释了六元，返工和报废解释了三元，加班和外协又解释了四元。库存周转天数从三十二天升到三十八天，供应商按时交付率为百分之九十二点八。铜端子还有两份报价未返回。为了客户审核，我们还需要完整的培训、能源消耗、投诉和供应商证明材料。",
    "role_meeting_v3_010": "短期内，我认为最确定的效果来自返工、无计划加班和慢动库存。我们可以用清晰的缺陷清单减少重复返工，只为已确认的订单批准额外班次。铜端子的价格不能单独看；供应商便宜但延迟交付会增加报废、加急运输和合同风险。八月份我会做一份从预算到实际单位成本的桥接表，并把立即有效的措施与必须等供应商确认后才能落实的措施分开。",
    "role_meeting_v3_021": "明白。我今天开始做对比模型，并标记所有仍需供应商或销售确认的信息。评估会包含价格、付款期限、安全库存、报废、物流和现金占用。九月八日前，我会为每个关键供应商提交一份综合建议。这个方案目前还不能做最终决策，但未决事项和下一次审核时间都会清楚记录。",
}


def _run_ffmpeg(ffmpeg: str, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000 or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"converted audio has wrong format: {path}")
        return round(handle.getnframes() / handle.getframerate(), 3)


def _load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="normal-speed full audio")
    parser.add_argument("--script", type=Path, required=True, help="v3 JSONL manuscript")
    parser.add_argument("--segments-dir", type=Path, required=True, help="normal-speed segment WAV directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/benchmarks/manufacturing_role_meeting_v3"),
    )
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert the 24 kHz source to 16 kHz PCM16")
    source_audio = args.audio.resolve()
    source_script = args.script.resolve()
    source_segments = args.segments_dir.resolve()
    output = args.output.resolve()
    if not source_audio.is_file():
        raise FileNotFoundError(source_audio)
    records = _load_records(source_script)

    full_target = output / "audio" / "full_normal_random_pauses_16k.wav"
    _run_ffmpeg(ffmpeg, source_audio, full_target)
    full_duration = _duration(full_target)

    samples = []
    for record in records:
        segment_id = str(record["segment_id"])
        source_segment = source_segments / f"{segment_id}.wav"
        if not source_segment.is_file():
            raise FileNotFoundError(source_segment)
        target = output / "audio" / "segments" / f"{segment_id}.wav"
        _run_ffmpeg(ffmpeg, source_segment, target)
        language, variant, group = LANGUAGE_MAP[record["language"]]
        sample = {
            "sample_id": segment_id,
            "audio_path": str(target.relative_to(output)).replace("\\", "/"),
            "duration_seconds": _duration(target),
            "language": language,
            "speech_variant": variant,
            "group": group,
            "scenario": record.get("phase", record.get("topic", "meeting")),
            "speaker_name": record.get("speaker_name"),
            "reference_text": record["text"],
            "text": record["text"],
            "reference_translation": TRANSLATIONS.get(segment_id),
            "translation": TRANSLATIONS.get(segment_id),
            "turn_count": 1,
            "speaker_count": 1,
            "reference_language_switches": 0,
            "decode_language_hint": True,
        }
        samples.append(sample)

    manifest = {
        "schema_version": "2.0-generated",
        "recording_seconds": full_duration,
        "full_audio_path": str(full_target.relative_to(output)).replace("\\", "/"),
        "source_audio": str(source_audio),
        "source_script": str(source_script),
        "source_segments": str(source_segments),
        "samples": samples,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {full_target}")
    print(f"samples={len(samples)}, recording_seconds={full_duration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

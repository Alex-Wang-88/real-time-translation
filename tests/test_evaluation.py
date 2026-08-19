from __future__ import annotations

from realtime_meeting.evaluation import (
    align_samples,
    evaluate_postprocess_api,
    evaluate_realtime_replay,
    score_text,
)


def test_score_text_uses_cjk_characters_and_latin_words() -> None:
    chinese = score_text("项目按计划推进", "项目按计划推进", "zh")
    english = score_text("ship the beta friday", "ship the beta", "en")

    assert chinese["reference_tokens"] == 7
    assert chinese["edit_distance"] == 0
    assert english["reference_tokens"] == 4
    assert english["edit_distance"] == 1
    assert english["error_rate"] == 0.25


def test_alignment_allows_one_reference_turn_to_split_into_two_paragraphs() -> None:
    samples = [
        {
            "sample_id": "s1",
            "language": "de",
            "speech_variant": "unknown",
            "duration_seconds": 10.0,
            "reference_text": "Kosten liegen über dem Budget und die Lieferung ist verspätet.",
            "reference_translation": "成本高于预算且交付延迟。",
        },
        {
            "sample_id": "s2",
            "language": "zh",
            "speech_variant": "mandarin",
            "duration_seconds": 4.0,
            "reference_text": "请给出下个月的计划。",
        },
    ]
    paragraphs = [
        {
            "segment_id": "p-1",
            "start": 0.0,
            "end": 5.0,
            "language": "de",
            "speech_variant": None,
            "text": "Kosten liegen über dem Budget",
            "translation_zh": "成本高于预算",
            "translation_status": "ready",
        },
        {
            "segment_id": "p-2",
            "start": 5.0,
            "end": 10.0,
            "language": "de",
            "speech_variant": None,
            "text": "und die Lieferung ist verspätet",
            "translation_zh": "且交付延迟",
            "translation_status": "ready",
        },
        {
            "segment_id": "p-3",
            "start": 11.0,
            "end": 15.0,
            "language": "zh",
            "speech_variant": None,
            "text": "请给出下个月的计划",
            "translation_zh": "请给出下个月的计划",
            "translation_status": "not_needed",
        },
    ]

    entries, extras, _ = align_samples(samples, paragraphs)

    assert extras == []
    assert [entry["actual_indices"] for entry in entries] == [[0, 1], [2]]


def test_postprocess_api_contract_requires_one_completed_request() -> None:
    passed = {
        "request_count": 1,
        "request": {"request": {"status_code": 202}, "task": {"status": "complete"}},
        "final_state": {"summary_state": "complete", "todo_state": "complete", "summary_error": None, "todo_error": None, "agent_error": None},
    }
    failed = {**passed, "request": {**passed["request"], "request": {"status_code": 500}}}

    assert evaluate_postprocess_api(passed)["passed"] is True
    assert evaluate_postprocess_api(failed)["passed"] is False


def test_replay_evaluation_reports_translation_variant_and_contract_metrics() -> None:
    manifest = {
        "samples": [
            {
                "sample_id": "zh-1",
                "language": "zh",
                "speech_variant": "sichuan",
                "duration_seconds": 3.0,
                "reference_text": "这个方案今天先确认。",
            },
            {
                "sample_id": "en-1",
                "language": "en",
                "speech_variant": "unknown",
                "duration_seconds": 3.0,
                "reference_text": "We will confirm the date today.",
                "reference_translation": "我们今天确认日期。",
            },
        ]
    }
    report = {
        "recording_state": "complete",
        "runtime_metrics": {"stage_failures": 0},
        "paragraphs": [
            {"segment_id": "p-1", "start": 0, "end": 3, "language": "zh", "speech_variant": "sichuan", "text": "这个方案今天先确认。", "translation_status": "not_needed"},
            {"segment_id": "p-2", "start": 4, "end": 7, "language": "en", "speech_variant": None, "text": "We will confirm the date today.", "translation_zh": "我们今天确认日期。", "translation_status": "ready"},
        ],
        "postprocess_api": {
            "request_count": 1,
            "request": {"request": {"status_code": 202}, "task": {"status": "complete"}},
            "final_state": {"summary_state": "complete", "todo_state": "complete", "summary_error": None, "todo_error": None, "agent_error": None},
        },
    }

    evaluation = evaluate_realtime_replay(manifest, report)

    assert evaluation["contract"]["passed"] is True
    assert evaluation["summary"]["sichuan_variant_accuracy"] == 1.0
    assert evaluation["summary"]["translation_success_rate"] == 1.0
    assert evaluation["summary"]["asr_token_weighted_error_rate"] == 0.0


def test_sichuan_evaluation_keeps_surface_text_and_scores_optional_mandarin_text() -> None:
    manifest = {
        "evaluation_contract": {"postprocess_api_required": False},
        "samples": [
            {
                "sample_id": "sc-1",
                "language": "zh",
                "speech_variant": "sichuan",
                "duration_seconds": 2.0,
                "text_sichuan": "莫得问题，明天给你回话",
                "text_mandarin": "没有问题，明天回复你",
            }
        ],
    }
    report = {
        "recording_state": "complete",
        "runtime_metrics": {"stage_failures": 0},
        "paragraphs": [
            {
                "segment_id": "p-1",
                "start": 0,
                "end": 2,
                "language": "zh",
                "speech_variant": "sichuan",
                "text": "莫得问题，明天给你回话",
                "mandarin_text": "没有问题，明天回复你",
                "translation_status": "not_needed",
            }
        ],
    }

    evaluation = evaluate_realtime_replay(manifest, report)

    assert evaluation["contract"]["passed"] is True
    assert evaluation["contract"]["postprocess_api_required"] is False
    assert evaluation["summary"]["sichuan_surface_error_rate"] == 0.0
    assert evaluation["summary"]["sichuan_mandarin_reference_samples"] == 1
    assert evaluation["summary"]["sichuan_mandarin_scored_samples"] == 1
    assert evaluation["segments"][0]["mandarin_semantic"]["status"] == "scored"

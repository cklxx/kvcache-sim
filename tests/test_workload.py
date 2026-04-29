import json

import pytest

from trace.workload import load_workload_trace, summarize_workload


def test_load_burstgpt_csv_skips_failed_rows_and_scales_to_ms(tmp_path) -> None:
    path = tmp_path / "burst.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,Session ID,Model,Request tokens,Response tokens,Total tokens,Log Type",
                "5,s1,ChatGPT,32,8,40,Conversation log",
                "45,s1,ChatGPT,64,0,64,Conversation log",
                "50,s2,GPT-4,16,4,20,API log",
            ]
        )
    )

    trace = load_workload_trace(
        path,
        block_size_bytes=1024,
        tokens_per_block=16,
    )

    assert trace.format_name == "burstgpt"
    assert trace.source_rows == 3
    assert trace.skipped_rows == 1
    assert [r.timestamp for r in trace.requests] == [0.0, 45_000.0]
    assert [r.output_tokens for r in trace.requests] == [8, 4]
    assert len(trace.requests[0].block_hashes) == 2
    assert trace.requests[0].block_hashes[0].startswith("sess:s1:ChatGPT")


def test_load_azure_datetime_trace(tmp_path) -> None:
    path = tmp_path / "azure.csv"
    path.write_text(
        "\n".join(
            [
                "TIMESTAMP,ContextTokens,GeneratedTokens",
                "2023-11-16 18:17:03.9799600,4808,10",
                "2023-11-16 18:17:04.0319600,3180,8",
            ]
        )
    )

    trace = load_workload_trace(
        path,
        block_size_bytes=1024,
        tokens_per_block=16,
    )

    assert trace.format_name == "azure"
    assert [r.timestamp for r in trace.requests] == pytest.approx([0.0, 52.0])
    assert trace.requests[0].prompt_tokens == 4808
    assert trace.requests[0].session_id.startswith("request_")


def test_load_mooncake_jsonl_preserves_hash_ids_and_hash_block_size(tmp_path) -> None:
    path = tmp_path / "mooncake.jsonl"
    rows = [
        {"timestamp": 3000, "input_length": 1024, "output_length": 7, "hash_ids": [0, 1]},
        {"timestamp": 6000, "input_length": 2048, "output_length": 9, "hash_ids": "[0,2,3]"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    trace = load_workload_trace(
        path,
        block_size_bytes=1024,
        kv_bytes_per_token=10,
        tokens_per_block=16,
    )
    summary = summarize_workload(trace)

    assert trace.format_name == "mooncake"
    assert trace.used_hash_ids
    assert trace.hash_tokens_per_block == 512
    assert [r.timestamp for r in trace.requests] == [0.0, 3000.0]
    assert trace.requests[0].block_hashes == ["mooncake:h:0", "mooncake:h:1"]
    assert trace.requests[0].block_size == 5120
    assert summary["hash_backed"] is True

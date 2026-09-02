#!/usr/bin/env bash

set -eu

mkdir -p traces

base_url="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release"

# the three FAST'25 traces live under traces/, the older arxiv one next to it
for trace_name in conversation_trace.jsonl toolagent_trace.jsonl synthetic_trace.jsonl; do
    target_path="traces/${trace_name}"

    if [ ! -f "$target_path" ]; then
        curl -sSL -o "$target_path" "$base_url/traces/$trace_name"
    fi
done

target_path="traces/mooncake_trace.jsonl"

if [ ! -f "$target_path" ]; then
    curl -sSL -o "$target_path" "$base_url/arxiv-trace/mooncake_trace.jsonl"
fi

wc -l traces/*.jsonl

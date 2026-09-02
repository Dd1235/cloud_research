import json

from .request import Request


# every hash id in the trace stands for one block of this many prompt tokens,
# so workers replaying it must be built with the same block size
MOONCAKE_BLOCK_SIZE = 512


def load_mooncake(
    path,
    *,
    speedup: float = 1.0,
    limit: int | None = None,
) -> list[Request]:
    """speedup > 1 compresses arrival gaps (raises load) without touching lengths.
    Block ids become ("m", hash_id) tuples so they never collide with the synthetic generator's ids.
    output_tokens is at least 1 (a request always emits a first token). Missing/empty hash_ids -> blocks=()."""
    requests = []

    with open(path) as trace_file:
        for line_index, line in enumerate(trace_file):
            if limit is not None and line_index >= limit:
                break

            trace_request = json.loads(line)
            hash_ids = trace_request.get("hash_ids") or []
            requests.append(
                Request(
                    id=line_index,
                    arrival=trace_request["timestamp"] / 1000.0 / speedup,
                    prompt_tokens=int(trace_request["input_length"]),
                    output_tokens=max(int(trace_request["output_length"]), 1),
                    blocks=tuple(("m", hash_id) for hash_id in hash_ids),
                ),
            )

    return requests

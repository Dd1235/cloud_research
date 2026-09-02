from typing import Protocol


class Policy(Protocol):

    # every policy is one method: look at the request and the workers, pick one.
    # keeping it this narrow is what lets the go router mirror the same shape later
    # (llm-d's filter -> score -> pick collapses into this).
    def choose(self, req, workers):
        ...

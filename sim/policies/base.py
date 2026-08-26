from typing import Protocol


class Policy(Protocol):

    def choose(self, req, workers):
        
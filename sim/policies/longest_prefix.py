class LongestPrefix:
    name = "longest_prefix"

    def choose(self, req, workers):
        return max(
            workers,
            key=lambda worker: (
                worker.view.match(req.blocks),
                -worker.outstanding,
            ),
        )
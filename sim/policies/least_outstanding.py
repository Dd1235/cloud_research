class LeastOutstanding:
    name = "least_outstanding"

    def choose(self, req, workers):
        return min(
            workers,
            key=lambda worker: worker.outstanding,
        )
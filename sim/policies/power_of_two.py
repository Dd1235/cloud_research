class PowerOfTwo:
    name = "p2c"

    def __init__(self, rng):
        self.rng = rng

    def choose(self, req, workers):
        if len(workers) < 2:
            return workers[0]

        i, j = self.rng.choice(
            len(workers),
            size=2,
            replace=False,
        )

        first = workers[i]
        second = workers[j]

        first_score = (first.outstanding, first.id)
        second_score = (second.outstanding, second.id)

        return (
            first
            if first_score <= second_score
            else second
        )
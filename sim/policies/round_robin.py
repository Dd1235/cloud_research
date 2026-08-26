class RoundRobin:

    name = "round_robin"

    def __init__(self):
        self._next_index = 0


    def choose(self, req, workers):

        worker = workers[self._next_index % len(workers)]
        self._next_index += 1
        return worker
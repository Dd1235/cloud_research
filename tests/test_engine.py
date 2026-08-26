from sim.engine import Engine


def test_events_run_in_time_order_and_fifo_on_times():

    engine = Engine(seed=0)
    log = []

    engine.schedule(2.0, log.append, "c")
    engine.schedule(1.0, log.append, "a")
    engine.schedule(1.0, log.append, "b")

    engine.run()

    assert log == ["a", "b", "c"]
    assert engine.now == 2.0

def test_run_until_stops_early_and_keeps_future_events():
    
    engine = Engine()
    log = []

    engine.schedule(1.0, log.append, 1)
    engine.schedule(5.0, log.append, 5)

    engine.run(until=2.0)

    assert log == [1]
    assert engine.pending == 1

    engine.run()

    assert log == [1, 5]
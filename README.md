# prefix-serve

Prefix-aware LLM serving control plane — a discrete-event simulator first, a real router later.

```
uv sync          # create venv + install deps
uv run pytest    # run tests
```

Layout (grows one commit at a time): `sim/` simulator package · `scripts/` experiment drivers · `tests/`.

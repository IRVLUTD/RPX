# Contributing

Developer docs live under `benchmark/docs/guides/contributing.md` and
on the published site at <https://irvlutd.github.io/RPX/guides/contributing/>.

## TL;DR

```bash
cd benchmark
make install          # editable install + pre-commit hooks
make check            # lint + type + coverage (what CI runs)
make smoke            # runnable end-to-end example
```

See `benchmark/Makefile` for every target (`make help`).

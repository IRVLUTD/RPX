
## Update Protocol (RPX Benchmark)

Every change to the benchmark codebase must update **all four artifacts** in the same commit:

1. **Code** — the implementation in `rpx_benchmark/`
2. **Docs** — the relevant markdown in `docs/methods/` or `docs/`
3. **PDF** — regenerate the PDF via pandoc
4. **SHARED_CONTEXT.md** — append a timestamped entry so the parallel session stays in sync

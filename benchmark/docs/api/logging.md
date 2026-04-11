# Logging (`rpx_benchmark.logging_utils`)

Thin wrappers over the stdlib `logging` module. Library modules
call `get_logger(__name__)` at module top; the CLI entrypoint calls
`configure_logging(level)` once to install a handler (rich when
available, plain otherwise).

::: rpx_benchmark.logging_utils
    options:
      show_root_toc_entry: false

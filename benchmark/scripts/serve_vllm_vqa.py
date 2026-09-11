#!/usr/bin/env python3
"""Keep one VQA model resident and expose its adapter over localhost HTTP."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from vqa_models.backend_registry import CHECKPOINTS, create_runner, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    runner = create_runner(
        args.model,
        args.image_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    checkpoint = CHECKPOINTS[args.model]
    runtime = provenance(args.model)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json(404, {"error": "not found"})
                return
            self._json(
                200,
                {
                    "status": "ready",
                    "model": args.model,
                    **runtime,
                    "checkpoint": checkpoint.repo_id,
                    "revision": checkpoint.revision,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                raw = runner.predict(
                    request["image_paths"],
                    request["prompt"],
                    int(request["max_tokens"]),
                    request["output_kind"],
                )
                latency_ms = (time.perf_counter() - started) * 1000
                self._json(
                    200,
                    {
                        "raw_output": raw,
                        "latency_ms": latency_ms,
                        "adapter_metadata": runner.prediction_metadata(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": str(exc), "error_type": type(exc).__name__})

        def log_message(self, fmt: str, *values: object) -> None:
            print(f"client={self.client_address[0]} {fmt % values}", flush=True)

    print(
        json.dumps(
            {
                "status": "ready",
                "model": args.model,
                "checkpoint": checkpoint.repo_id,
                "revision": checkpoint.revision,
                "host": args.host,
                "port": args.port,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

"""Custom VQA callables share the official parser and denominator."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from PIL import Image

from rpx_benchmark.exceptions import ConfigError
from rpx_benchmark.vqa.contract import VQASample, write_manifest
from rpx_benchmark.vqa.hub_rgb import image_cache_name

evaluate = importlib.import_module("rpx_benchmark.vqa.evaluate")


def fixture_manifest(root):
    samples = [VQASample.from_dict(dict(
        scene_id="scene001", kind="mos", phase=0, frame=f"{index:05d}",
        img_w=100, img_h=100, type="attr_single_color", question="Which object is red?",
        answer="secret_ground_truth", answer_bbox=[0, 0, 99, 99],
    )) for index in range(3)]
    manifest = root / "samples.jsonl"
    write_manifest(samples, manifest)
    cache = root / "images"
    cache.mkdir()
    for sample in samples:
        Image.new("RGB", (100, 100), "red").save(cache / image_cache_name(sample))
    return manifest, cache


def test_callable_failures_remain_in_denominator_and_raw_outputs(tmp_path):
    manifest, cache = fixture_manifest(tmp_path)
    calls = []
    def predict(paths, prompt, max_tokens, output_kind):
        assert len(paths) == 1 and Path(paths[0]).exists()
        assert "secret_ground_truth" not in prompt
        assert max_tokens > 0 and output_kind.startswith("bbox")
        calls.append(paths)
        if len(calls) == 1:
            return '{"label":"cup","bbox":[0,0,1000,1000]}'
        if len(calls) == 2:
            return 'malformed response'
        raise TimeoutError("API timed out")
    out = tmp_path / "results"
    result = evaluate.evaluate_vqa(manifest, predict, model_name="my-model", model_revision="test-v1",
                                   output_dir=out, image_cache=cache)
    assert result["num_samples"] == 3
    assert len(calls) == 3
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [r["parsed"]["valid"] for r in rows] == [True, False, False]
    assert rows[-1]["inference_error"] == "TimeoutError: API timed out"
    assert result["metrics"]["bbox_accuracy_at_0_5"] == pytest.approx(1 / 3)
    assert json.loads((out / "result.json").read_text()) == result
    with pytest.raises(ConfigError, match="empty output"):
        evaluate.evaluate_vqa(manifest, predict, model_name="my-model", model_revision="test-v1",
                              output_dir=out, image_cache=cache)


def test_two_image_order_is_preserved_without_exposing_answers(tmp_path, monkeypatch):
    manifest, cache = fixture_manifest(tmp_path)
    samples = evaluate.load_manifest(manifest)
    # Fetch/crop/hash behavior is covered by the hub_rgb contract tests; here
    # exercise the public callable boundary and preserve its ordered images.
    reference, target = cache / "reference.png", cache / "target.png"
    Image.new("RGB", (4, 4)).save(reference)
    Image.new("RGB", (100, 100)).save(target)
    monkeypatch.setattr(evaluate, "load_manifest", lambda path: samples[:1])
    monkeypatch.setattr(evaluate, "fetch_images", lambda sample, cache: (reference, target))
    def predict(paths, prompt, max_tokens, output_kind):
        assert paths == (reference, target)
        return '{"bbox":[0,0,1000,1000]}'
    result = evaluate.evaluate_vqa(manifest, predict, model_name="custom", model_revision="1",
                                   output_dir=tmp_path / "two", image_cache=cache)
    assert result["num_samples"] == 1


def test_custom_model_over_http(tmp_path):
    """Exercise a real HTTP call through the public callable API on localhost."""
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    manifest, cache = fixture_manifest(tmp_path)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            data = json.dumps({"text": '{"bbox":[0,0,1000,1000]}'}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def predict(paths, prompt, max_tokens, output_kind):
            import base64
            payload = {"prompt": prompt, "images": [base64.b64encode(p.read_bytes()).decode() for p in paths],
                       "max_tokens": max_tokens, "output_kind": output_kind}
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/predict",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)["text"]
        report = evaluate.evaluate_vqa(manifest, predict, model_name="http-model", model_revision="v1",
                                       output_dir=tmp_path / "http", image_cache=cache)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(requests) == 3
    assert all(len(request["images"]) == 1 for request in requests)
    assert all("secret_ground_truth" not in request["prompt"] for request in requests)
    assert report["metrics"]["bbox_accuracy_at_0_5"] == 1.0

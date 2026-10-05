"""Keep the runnable extension guides aligned with the registration APIs."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("guide", ["metrics", "tasks"])
def test_extension_guide_example(guide):
    import rpx_benchmark
    from rpx_benchmark.api import TaskType
    from rpx_benchmark.metrics import available_metrics
    from rpx_benchmark.tasks.registry import get_task_spec

    root = Path(rpx_benchmark.__file__).parent / "guides"
    text = (root / guide / "README.md").read_text()
    example = re.findall(r"```python\n(.*?)```", text, re.DOTALL)[0]
    task = TaskType.MONOCULAR_DEPTH
    original = get_task_spec(task)
    metrics = available_metrics()
    exec(compile(example, str(root / guide / "README.md"), "exec"), {})
    assert get_task_spec(task) is original
    assert available_metrics() == metrics


def test_get_started_example(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    import numpy as np

    import rpx_benchmark
    from rpx_benchmark.api import TaskType
    from tests.test_user_workflows import local_case

    _, manifest = local_case(tmp_path, TaskType.MONOCULAR_DEPTH)
    assert manifest.name == "manifest.json"
    model_module = ModuleType("my_model")
    model_module.predict_depth = lambda rgb: np.full(rgb.shape[:2], 2.0, np.float32)
    monkeypatch.setitem(sys.modules, "my_model", model_module)
    monkeypatch.chdir(tmp_path)
    text = (Path(rpx_benchmark.__file__).parent / "guides/getting-started/README.md").read_text()
    example = re.findall(r"```python\n(.*?)```", text, re.DOTALL)[1]
    namespace = {}
    exec(compile(example, "getting-started/README.md", "exec"), namespace)
    assert namespace["result"].num_samples == 1
    assert namespace["result"].aggregated["absrel"] == 0.0

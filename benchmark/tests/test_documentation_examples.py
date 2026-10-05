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

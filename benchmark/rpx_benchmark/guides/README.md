# RPX Toolkit

**Evaluate perception on the same scenes, before, during, and after interaction.**

RPX provides dataset loaders, model adapters, task-specific metrics, and reports for robot perception. Use the reference model runners or connect your own local model or API. Model weights remain separate from the Python package.

<div class="grid cards" markdown>

-   **Run your first benchmark**

    Install the toolkit, load a manifest, and score a model.

    [Get started →](getting-started/README.md)

-   **Bring your own model**

    Wrap a NumPy callable or implement the adapter protocol.

    [Model integration →](models/README.md)

-   **Add a metric**

    Register a calculator and verify it on known predictions.

    [Metric guide →](metrics/README.md)

-   **Add a task**

    Define inputs, ground truth, scoring, and an evaluation runner.

    [Task guide →](tasks/README.md)

</div>

## How benchmarking works

A versioned manifest selects samples and identifies their scene, phase, view, and ground truth. A model adapter prepares the input and converts outputs to the toolkit's prediction types. Metrics score predictions against ground truth, and the runner saves per-sample and aggregate results.

```text
Manifest → Dataset loader → Model adapter → Predictions → Metrics → Reports
```

## Paper tasks

| Task | Input | Evaluation |
| --- | --- | --- |
| T1 · Image depth | RGB image | Metric-depth accuracy |
| T2 · Video depth | RGB sequence | Depth accuracy and temporal consistency |
| T3 · Object tracking | Video and initialization | Multi-object tracking and identity consistency |
| T4 · Relative camera pose | Image pairs | Relative-pose errors and threshold AUC |
| T5 · Visual grounding | Image and text question | Localization accuracy |
| T6 · In-context grounding | Image, question, and reference image | Reference-conditioned localization |

T5 and T6 use visual-grounding tooling with distinct evaluation protocols. The toolkit additionally exposes detection, segmentation, sparse depth, and keypoint matching; see `available_tasks()` for registered runners.

## Concepts and acronyms

- **MOS / SOS:** multi-object scenes / single-object captures.
- **Phase:** Clutter, Interaction, or Clean. Egocentric Interaction is a separate view.
- **ESD:** empirical scene difficulty, used to group scenes by difficulty.
- **J / Jmin:** normalized desirability and worst-phase quality.
- **Φ:** phase robustness, computed by the paper analysis tools; it is not the mean of raw metric values.
- **GT:** ground truth. Use consistent units, coordinates, and object identities across predictions and GT.

[Browse the API reference](api/README.md), [read the repository overview](https://github.com/IRVLUTD/RPX), or [install from PyPI](https://pypi.org/project/rpx-benchmark/).

The toolkit is released under the MIT license. Dataset and pretrained model licenses must be checked separately.

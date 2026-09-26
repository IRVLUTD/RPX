"""Opt-in measurement hooks that leave each image's native configuration intact.

Used only by profile_vision.py in a disposable profiling process. Never copy
an old runner/config/registry onto a newer image to install instrumentation.
"""

from __future__ import annotations

import importlib
import inspect
from contextlib import ExitStack

from rpx_benchmark.exceptions import ConfigError
from rpx_benchmark.hardware_profile import HardwareProfile, batch_metadata


class RuntimeHooks:
    def __init__(self, config):
        self.config = config
        self.stack = ExitStack()
        self.profiles = []

    def replace(self, obj, name, replacement):
        old = getattr(obj, name)
        self.stack.callback(setattr, obj, name, old)
        setattr(obj, name, replacement)

    def profile(self, model):
        profile = HardwareProfile(model, self.config)
        self.profiles.append(profile)
        return profile

    def close(self):
        self.stack.close()

    def finish(self):
        if self.profiles:
            self.profiles[-1].finish()

    def install(self):
        task = self.config["task"]
        if task in {"image-depth", "rcpe"}:
            module = importlib.import_module("rpx_benchmark.runner")

            def run_with_report(runner, *args, **kwargs):
                if runner.call_setup:
                    runner.model.setup()
                profile = self.profile(runner.model)
                for batch in runner.dataset:
                    if not batch:
                        continue
                    profile.call(
                        lambda batch=batch: runner.model.predict(batch),
                        units=len(batch),
                        metadata=batch_metadata(batch),
                        cache_hits=lambda: getattr(runner.model, "last_cache_hits", None),
                    )
                profile.finish()

            self.replace(module.BenchmarkRunner, "run_with_report", run_with_report)
        elif task == "video-depth":
            module = importlib.import_module("rpx_benchmark.tasks._video_pipeline")
            entrypoint = importlib.import_module("rpx_benchmark.tasks.video_depth")
            original = module.run_video_pipeline
            signature = inspect.signature(original)

            def pipeline(*args, **kwargs):
                cfg = signature.bind(*args, **kwargs).arguments["cfg"]
                model = cfg.model
                original_predict = model.predict
                profile = None

                def predict(batch, *predict_args, **predict_kwargs):
                    nonlocal profile
                    # Native pipeline performs setup before its first predict.
                    if profile is None:
                        profile = self.profile(model)
                    return profile.call(
                        lambda: original_predict(batch, *predict_args, **predict_kwargs),
                        units=sum(len(sample.rgb_seq) for sample in batch),
                        metadata=batch_metadata(batch),
                        cache_hits=lambda: getattr(model, "last_cache_hits", None),
                    )

                self.replace(model, "predict", predict)
                return original(*args, **kwargs)

            self.replace(module, "run_video_pipeline", pipeline)
            # This module imports the function by value in existing images.
            if getattr(entrypoint, "run_video_pipeline", None) is original:
                self.replace(entrypoint, "run_video_pipeline", pipeline)
        elif task == "tracking":
            registry = importlib.import_module("tracking_models").TRACKER_CLASSES
            name = self.config["model"]
            if name not in registry:
                raise ConfigError(
                    f"Image tracking registry does not contain {name!r}; available: {sorted(registry)}"
                )
            cls = registry[name]
            original = cls.track
            signature = inspect.signature(original)
            instances = {}

            def track(tracker, *args, **kwargs):
                bound = signature.bind(tracker, *args, **kwargs)
                bound.apply_defaults()
                units = bound.arguments.get("frame_count")
                if not isinstance(units, int) or units < 1:
                    raise ConfigError(
                        "Native tracking adapter does not expose a positive frame_count"
                    )
                if id(tracker) not in instances:
                    instances[id(tracker)] = self.profile(tracker)
                prompt_type = getattr(tracker, "prompt_type", None)
                return instances[id(tracker)].call(
                    lambda: original(tracker, *args, **kwargs),
                    units=units,
                    metadata={
                        "video_dir": str(bound.arguments.get("video_dir", "")),
                        "initialization": prompt_type,
                        "initialization_frames_included": 1
                        if prompt_type in {"box", "bbox", "mask"}
                        else 0,
                    },
                )

            self.replace(cls, "track", track)
        else:
            raise ConfigError(f"Unsupported hardware profiling task: {task}")
        return self

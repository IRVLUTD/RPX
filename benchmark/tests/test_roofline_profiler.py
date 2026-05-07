"""Tests for Tier 1/2/3 profiler additions: roofline, system card, traffic."""

from __future__ import annotations

import pytest

from rpx_benchmark.profiler import (
    REFERENCE_GPUS,
    EfficiencyMetadata,
    GPUSpec,
    RooflineBound,
    SystemCard,
)

# --------------------------------------------------------------------------- #
# GPUSpec + RooflineBound
# --------------------------------------------------------------------------- #


class TestRooflineBound:
    """Roofline latency bound computation."""

    def test_compute_bound_model(self) -> None:
        """A model with high FLOPs and low traffic should be compute-bound."""
        gpu = GPUSpec(name="TestGPU", peak_tflops=10.0, memory_bw_gbps=1000.0, memory_gb=24.0)
        bound = RooflineBound.from_model_and_gpu(
            flops_g=100.0,  # 100 GFLOP
            traffic_gb=0.01,  # 10 MB — very low traffic
            gpu=gpu,
        )
        assert bound.bottleneck == "compute"
        assert bound.latency_ms == bound.compute_ms
        # 100 GFLOP / (10 TFLOP/s) = 100e9 / 10e12 = 0.01s = 10ms
        assert bound.compute_ms == pytest.approx(10.0, rel=1e-3)

    def test_memory_bound_model(self) -> None:
        """A model with low FLOPs and high traffic should be memory-bound."""
        gpu = GPUSpec(name="TestGPU", peak_tflops=100.0, memory_bw_gbps=100.0, memory_gb=24.0)
        bound = RooflineBound.from_model_and_gpu(
            flops_g=1.0,  # 1 GFLOP — tiny compute
            traffic_gb=1.0,  # 1 GB — lots of traffic
            gpu=gpu,
        )
        assert bound.bottleneck == "memory"
        assert bound.latency_ms == bound.memory_ms
        # 1 GB / 100 GB/s = 0.01s = 10ms
        assert bound.memory_ms == pytest.approx(10.0, rel=1e-3)

    def test_reference_gpus_exist(self) -> None:
        """All three reference GPUs should be populated."""
        assert "A100-80GB" in REFERENCE_GPUS
        assert "RTX 4090" in REFERENCE_GPUS
        assert "Jetson Orin 64GB" in REFERENCE_GPUS

    def test_orin_slower_than_a100(self) -> None:
        """Jetson Orin should always predict slower than A100."""
        a100 = REFERENCE_GPUS["A100-80GB"]
        orin = REFERENCE_GPUS["Jetson Orin 64GB"]
        b_a100 = RooflineBound.from_model_and_gpu(50.0, 1.0, a100)
        b_orin = RooflineBound.from_model_and_gpu(50.0, 1.0, orin)
        assert b_orin.latency_ms > b_a100.latency_ms


# --------------------------------------------------------------------------- #
# EfficiencyMetadata — Tier 1 derivation
# --------------------------------------------------------------------------- #


class TestEfficiencyMetadataTier1:
    """Tier 1: derived fields (MACs, traffic, arithmetic intensity)."""

    def test_derive_macs_from_flops(self) -> None:
        em = EfficiencyMetadata(flops_g=100.0)
        em.derive_tier1()
        assert em.macs_g == pytest.approx(50.0)

    def test_derive_traffic_from_params(self) -> None:
        em = EfficiencyMetadata(params_m=100.0)  # 100M params
        em.derive_tier1()
        # 100M * 4B * 3 / 1e9 = 1.2 GB
        assert em.memory_traffic_gb == pytest.approx(1.2, rel=1e-2)

    def test_derive_arithmetic_intensity(self) -> None:
        em = EfficiencyMetadata(flops_g=100.0, params_m=100.0)
        em.derive_tier1()
        # AI = 100e9 / 1.2e9 ≈ 83.33
        assert em.arithmetic_intensity is not None
        assert em.arithmetic_intensity > 0

    def test_derive_does_not_overwrite_explicit(self) -> None:
        """If user provides traffic, derive should not overwrite it."""
        em = EfficiencyMetadata(
            flops_g=100.0,
            params_m=100.0,
            memory_traffic_gb=2.5,  # explicit
        )
        em.derive_tier1()
        assert em.memory_traffic_gb == 2.5  # NOT overwritten
        # AI = 100e9 / 2.5e9 = 40
        assert em.arithmetic_intensity == pytest.approx(40.0, rel=1e-2)

    def test_derive_does_not_overwrite_macs(self) -> None:
        em = EfficiencyMetadata(flops_g=100.0, macs_g=42.0)
        em.derive_tier1()
        assert em.macs_g == 42.0


class TestEfficiencyMetadataRoofline:
    """Tier 2: roofline bound computation via EfficiencyMetadata."""

    def test_compute_roofline_default_gpus(self) -> None:
        em = EfficiencyMetadata(flops_g=50.0, params_m=300.0)
        bounds = em.compute_roofline()
        assert len(bounds) == 3
        assert "A100-80GB" in bounds
        assert "RTX 4090" in bounds
        assert "Jetson Orin 64GB" in bounds
        for _name, b in bounds.items():
            assert b.latency_ms > 0
            assert b.bottleneck in ("compute", "memory")

    def test_compute_roofline_custom_gpu(self) -> None:
        em = EfficiencyMetadata(flops_g=10.0, params_m=50.0)
        custom = GPUSpec(name="MyGPU", peak_tflops=5.0, memory_bw_gbps=500.0, memory_gb=8.0)
        bounds = em.compute_roofline(specs=[custom])
        assert "MyGPU" in bounds

    def test_compute_roofline_missing_flops_returns_empty(self) -> None:
        em = EfficiencyMetadata(params_m=100.0)
        bounds = em.compute_roofline()
        assert bounds == {}

    def test_roofline_stored_on_metadata(self) -> None:
        em = EfficiencyMetadata(flops_g=50.0, params_m=300.0)
        em.compute_roofline()
        assert em.roofline is not None


class TestEfficiencyMetadataTableRow:
    """to_table_row includes new fields."""

    def test_table_row_includes_tier1(self) -> None:
        em = EfficiencyMetadata(flops_g=50.0, params_m=100.0)
        em.derive_tier1()
        row = em.to_table_row()
        assert "macs_g" in row
        assert "memory_traffic_gb" in row
        assert "arithmetic_intensity" in row
        assert row["macs_g"] == pytest.approx(25.0)

    def test_table_row_includes_roofline(self) -> None:
        em = EfficiencyMetadata(flops_g=50.0, params_m=100.0)
        em.compute_roofline()
        row = em.to_table_row()
        # Should have roofline keys for each reference GPU
        assert any(k.startswith("roofline_") for k in row)
        assert any(k.startswith("bottleneck_") for k in row)

    def test_table_row_includes_system_card(self) -> None:
        em = EfficiencyMetadata(flops_g=50.0)
        em.system_card = SystemCard(gpu_name="TestGPU", precision="fp32")
        row = em.to_table_row()
        assert "system_card" in row
        assert row["system_card"]["gpu_name"] == "TestGPU"

    def test_api_model_table_row(self) -> None:
        em = EfficiencyMetadata(model_type="api")
        row = em.to_table_row()
        assert row["params_m"] == "N/A (API)"
        assert row["flops_g"] == "N/A (API)"


# --------------------------------------------------------------------------- #
# SystemCard
# --------------------------------------------------------------------------- #


class TestSystemCard:
    """SystemCard auto-detection and serialization."""

    def test_auto_detect_fills_python_version(self) -> None:
        card = SystemCard.auto_detect()
        assert card.python_version != ""
        assert "." in card.python_version  # e.g. "3.11.9"

    def test_auto_detect_fills_os(self) -> None:
        card = SystemCard.auto_detect()
        assert card.os != ""

    def test_to_dict_round_trip(self) -> None:
        card = SystemCard(
            gpu_name="RTX 4090",
            gpu_memory_gb=24.0,
            precision="fp16",
        )
        d = card.to_dict()
        assert d["gpu_name"] == "RTX 4090"
        assert d["gpu_memory_gb"] == 24.0
        assert d["precision"] == "fp16"

    def test_summary_string(self) -> None:
        card = SystemCard(
            gpu_name="A100",
            gpu_memory_gb=80.0,
            pytorch_version="2.5.0",
            cuda_version="12.4",
            precision="fp32",
        )
        s = card.summary()
        assert "A100" in s
        assert "80.0GB" in s
        assert "fp32" in s

    def test_auto_detect_precision_passthrough(self) -> None:
        card = SystemCard.auto_detect(precision="bf16")
        assert card.precision == "bf16"

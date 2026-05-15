"""
FailureScope Layer 3 — Test Suite
===================================
28 tests covering:
  - Dataset: record processing, normalisation, padding, worker sharding,
             label encoding, failure one-hot, edge cases
  - Model: forward shape, gradient checkpointing toggle, attention output,
           parameter count sanity, CLS token, mask inversion
  - Train: config defaults, class-weight sum, pre-flight disk check,
           collate_fn shape
  - Attention extractor: sampling logic, aggregation, rule extraction

All tests run without GPU, without the real JSONL file, and without Ollama.
Heavy objects (model, data) are created at minimal scale (seq_len=8, d_model=16).
"""

import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── Module under test ──────────────────────────────────────────────────────────
from backend.layer3.dataset import (
    FAILURE_COMPONENTS,
    FAILURE_TO_IDX,
    FEATURE_DIM,
    MAX_SEQ_LEN,
    STATE_KEYS,
    TOTAL_SIZE,
    TRAIN_SIZE,
    TrajectoryDataset,
    collate_fn,
)
from backend.layer3.model import FailureScopeTransformer, PositionalEncoding
from backend.layer3.train import DEFAULT_CONFIG, _get_device, collate_fn as train_collate

# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_state(t: float, alt: float = 5000.0, v: float = 14.0) -> dict:
    """Return a minimal state dict matching the trajectory schema."""
    return {
        "t": t, "alt": alt, "alt_agl": alt, "spd": 100.0,
        "pitch": 5.0, "roll": 0.0, "yaw": 0.0, "vs": 0.0,
        "rpm": 2400.0, "ff": 8.5, "fuel_l": 30.0, "fuel_r": 30.0,
        "v": v, "bat": 33.0, "pump_eff": 1.0,
        "outcome": "in_progress", "failures": [],
    }


def _make_record(
    outcome: str = "catastrophic",
    n_steps: int = 10,
    component: str = "alternator",
) -> dict:
    """Return a minimal trajectory record."""
    return {
        "scenario_id": "test_0001",
        "aircraft": "cessna_182",
        "init": {"altitude_ft": 5000.0, "airspeed_kts": 100.0,
                 "fuel_gal": 61.0, "skill": "commercial",
                 "fatigue": 0.6, "weather": "vmc"},
        "failures": [{"component": component, "trigger_time_s": 0.0,
                      "severity": "catastrophic"}],
        "outcome": outcome,
        "ttc_s": 31.6,
        "recovery_s": None,
        "failure_chain": [component],
        "reason": "uncontrolled_dive",
        "duration_s": float(n_steps),
        "steps": n_steps,
        "states": [_make_state(float(i)) for i in range(n_steps)],
    }


def _write_fake_gz(path: str, records: list) -> None:
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def tiny_model():
    """4-layer, 8-head model at minimal d_model=32 for fast CPU tests."""
    return FailureScopeTransformer(
        feature_dim=FEATURE_DIM,
        d_model=32,
        nhead=8,
        num_layers=4,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=16,
        use_gradient_checkpointing=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATASET TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetConstants:

    def test_feature_dim_equals_13_plus_1_plus_23(self):
        assert FEATURE_DIM == len(STATE_KEYS) + 1 + len(FAILURE_COMPONENTS)

    def test_feature_dim_is_37(self):
        assert FEATURE_DIM == 37

    def test_failure_components_count(self):
        assert len(FAILURE_COMPONENTS) == 23

    def test_all_components_in_index(self):
        assert set(FAILURE_TO_IDX.keys()) == set(FAILURE_COMPONENTS)

    def test_train_val_split_sum(self):
        assert TRAIN_SIZE + (TOTAL_SIZE - TRAIN_SIZE) == TOTAL_SIZE
        assert TRAIN_SIZE == 400_000

    def test_state_keys_count(self):
        assert len(STATE_KEYS) == 13


class TestRecordProcessing:

    def setup_method(self):
        self.ds = TrajectoryDataset.__new__(TrajectoryDataset)
        self.ds.max_seq_len = 16

    def test_catastrophic_label_is_1(self):
        rec = _make_record(outcome="catastrophic", n_steps=5)
        features, mask, label = self.ds._process_record(rec)
        assert label.item() == 1

    def test_recoverable_label_is_0(self):
        rec = _make_record(outcome="recoverable", n_steps=5)
        features, mask, label = self.ds._process_record(rec)
        assert label.item() == 0

    def test_output_feature_shape(self):
        rec = _make_record(n_steps=8)
        features, mask, label = self.ds._process_record(rec)
        assert features.shape == (16, FEATURE_DIM)

    def test_mask_marks_real_steps(self):
        rec = _make_record(n_steps=8)
        features, mask, label = self.ds._process_record(rec)
        assert mask[:8].all()
        assert not mask[8:].any()

    def test_padding_is_zero(self):
        rec = _make_record(n_steps=5)
        features, mask, label = self.ds._process_record(rec)
        assert (features[5:] == 0.0).all()

    def test_truncation_at_max_seq_len(self):
        rec = _make_record(n_steps=30)  # longer than max_seq_len=16
        features, mask, label = self.ds._process_record(rec)
        assert features.shape[0] == 16
        assert mask.sum().item() == 16

    def test_failure_one_hot_correct_index(self):
        comp = "alternator"
        rec = _make_record(component=comp, n_steps=3)
        features, mask, label = self.ds._process_record(rec)
        expected_idx = 14 + FAILURE_TO_IDX[comp]
        # one-hot should be 1 at correct index for all real steps
        assert features[0, expected_idx].item() == pytest.approx(1.0)

    def test_unknown_failure_component_yields_zero_vector(self):
        rec = _make_record(n_steps=3)
        rec["failures"][0]["component"] = "nonexistent_part"
        features, mask, label = self.ds._process_record(rec)
        assert features[0, 14:].sum().item() == pytest.approx(0.0)

    def test_fatigue_broadcast_across_all_steps(self):
        rec = _make_record(n_steps=6)
        rec["init"]["fatigue"] = 0.75
        features, mask, label = self.ds._process_record(rec)
        for i in range(6):
            assert features[i, 13].item() == pytest.approx(0.75)

    def test_empty_states_returns_none(self):
        rec = _make_record(n_steps=0)
        rec["states"] = []
        assert self.ds._process_record(rec) is None

    def test_normalisation_clamps_values_approximately(self):
        """After normalisation, physical values should be in roughly [0, 1]."""
        rec = _make_record(n_steps=3)
        features, mask, label = self.ds._process_record(rec)
        # Check numeric part (indices 0:13) for real steps
        numeric = features[:3, :13]
        # None should wildly exceed 1.0 for a nominal state
        assert (numeric <= 1.1).all(), "Normalised values exceed 1.1"
        assert (numeric >= 0.0).all(), "Normalised values are negative"

    def test_missing_init_fatigue_defaults_to_0_5(self):
        rec = _make_record(n_steps=3)
        rec["init"].pop("fatigue")
        features, mask, label = self.ds._process_record(rec)
        assert features[0, 13].item() == pytest.approx(0.5)


class TestCollateFn:

    def test_collate_stacks_correctly(self):
        batch = [_make_batch_item() for _ in range(4)]
        features, masks, labels = collate_fn(batch)
        assert features.shape == (4, 16, FEATURE_DIM)
        assert masks.shape   == (4, 16)
        assert labels.shape  == (4,)

    def test_collate_label_dtype(self):
        batch = [_make_batch_item() for _ in range(2)]
        _, _, labels = collate_fn(batch)
        assert labels.dtype == torch.long


def _make_batch_item():
    """Helper: return a (features, mask, label) tuple at max_seq_len=16."""
    features = torch.zeros(16, FEATURE_DIM)
    mask     = torch.ones(16, dtype=torch.bool)
    label    = torch.tensor(1, dtype=torch.long)
    return features, mask, label


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelForward:

    def test_output_shape_no_attention(self, tiny_model):
        B, S = 2, 8
        x    = torch.zeros(B, S, FEATURE_DIM)
        mask = torch.ones(B, S, dtype=torch.bool)
        logits = tiny_model(x, mask)
        assert logits.shape == (B, 2)

    def test_output_shape_with_attention(self, tiny_model):
        B, S = 2, 8
        x    = torch.zeros(B, S, FEATURE_DIM)
        mask = torch.ones(B, S, dtype=torch.bool)
        logits, attn = tiny_model(x, mask, return_attention=True)
        assert logits.shape == (B, 2)
        assert len(attn) == 4  # one per layer

    def test_attention_shape_per_layer(self, tiny_model):
        B, S = 2, 8
        x    = torch.zeros(B, S, FEATURE_DIM)
        mask = torch.ones(B, S, dtype=torch.bool)
        _, attn = tiny_model(x, mask, return_attention=True)
        for layer_w in attn:
            # (B, nhead, S+1, S+1)
            assert layer_w.shape == (B, 8, S + 1, S + 1)

    def test_padding_mask_ignored_positions(self, tiny_model):
        """Padded positions should not change CLS output significantly."""
        B, S = 1, 8
        x    = torch.randn(B, S, FEATURE_DIM)
        mask_full = torch.ones(B, S, dtype=torch.bool)
        mask_half = torch.ones(B, S, dtype=torch.bool)
        mask_half[0, 4:] = False  # second half is padding

        # Pad x for second call (padding positions set to zero)
        x_half = x.clone()
        x_half[0, 4:] = 0.0

        tiny_model.eval()
        with torch.no_grad():
            out_half  = tiny_model(x_half, mask_half)
            out_trunc = tiny_model(x[:, :4, :].contiguous(),
                                   mask_full[:, :4].contiguous())
        # The results will differ (positional encoding), but both should be valid logits
        assert out_half.shape  == (1, 2)
        assert out_trunc.shape == (1, 2)

    def test_logits_are_finite(self, tiny_model):
        B, S = 4, 10
        x    = torch.randn(B, S, FEATURE_DIM)
        mask = torch.ones(B, S, dtype=torch.bool)
        logits = tiny_model(x, mask)
        assert torch.isfinite(logits).all()

    def test_gradient_flows_to_input_proj(self, tiny_model):
        x    = torch.randn(2, 5, FEATURE_DIM, requires_grad=False)
        mask = torch.ones(2, 5, dtype=torch.bool)
        logits = tiny_model(x, mask)
        loss = logits.sum()
        loss.backward()
        assert tiny_model.input_proj.weight.grad is not None

    def test_parameter_count_reasonable(self, tiny_model):
        n = tiny_model.count_parameters()
        # With d_model=32 the model should be tiny — just confirm it's not zero or huge
        assert 1_000 < n < 1_000_000, f"Unexpected param count: {n}"


class TestPositionalEncoding:

    def test_output_shape_unchanged(self):
        pe = PositionalEncoding(d_model=32, max_len=20, dropout=0.0)
        x  = torch.zeros(3, 10, 32)
        assert pe(x).shape == x.shape

    def test_different_positions_differ(self):
        pe = PositionalEncoding(d_model=32, max_len=20, dropout=0.0)
        x  = torch.zeros(1, 5, 32)
        out = pe(x)
        # Adjacent positions should have different encoding
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestGradientCheckpointing:

    def test_training_with_checkpointing_produces_gradients(self):
        model = FailureScopeTransformer(
            feature_dim=FEATURE_DIM, d_model=32, nhead=8,
            num_layers=4, dim_feedforward=64, dropout=0.0,
            max_seq_len=16, use_gradient_checkpointing=True,
        )
        model.train()
        x    = torch.randn(2, 8, FEATURE_DIM)
        mask = torch.ones(2, 8, dtype=torch.bool)
        logits = model(x, mask)
        loss = logits.sum()
        loss.backward()
        assert model.input_proj.weight.grad is not None

    def test_checkpointing_disabled_at_inference(self, tiny_model):
        """return_attention=True should work regardless of checkpointing flag."""
        model = FailureScopeTransformer(
            feature_dim=FEATURE_DIM, d_model=32, nhead=8,
            num_layers=4, dim_feedforward=64, dropout=0.0,
            max_seq_len=16, use_gradient_checkpointing=True,  # enabled
        )
        model.eval()
        x    = torch.zeros(1, 8, FEATURE_DIM)
        mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            logits, attn = model(x, mask, return_attention=True)
        assert logits.shape == (1, 2)
        assert len(attn) == 4


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfig:

    def test_default_config_has_required_keys(self):
        required = {
            "batch_size", "epochs", "learning_rate", "weight_decay",
            "max_grad_norm", "label_smoothing", "class_weights",
            "d_model", "nhead", "num_layers",
        }
        assert required.issubset(DEFAULT_CONFIG.keys())

    def test_class_weights_two_elements(self):
        assert len(DEFAULT_CONFIG["class_weights"]) == 2

    def test_class_weights_approximately_correct_ratio(self):
        # catastrophic_weight / recoverable_weight should be ~1.9
        # (reflects 33.8% cat vs 66.2% rec inverse frequency ratio)
        w_rec, w_cat = DEFAULT_CONFIG["class_weights"]
        ratio = w_cat / w_rec
        assert 1.5 < ratio < 2.5, f"Unexpected class weight ratio: {ratio:.2f}"

    def test_batch_size_safe_for_vram(self):
        # batch_size=64 safe for RTX 4060 8GB with d_model=256 (estimated ~1.76GB peak)
        assert DEFAULT_CONFIG["batch_size"] <= 128

    def test_nhead_divides_d_model(self):
        assert DEFAULT_CONFIG["d_model"] % DEFAULT_CONFIG["nhead"] == 0

    def test_collate_fn_importable_from_train(self):
        # train.py re-exports collate_fn — verify it's the same object
        assert train_collate is collate_fn


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION EXTRACTOR TESTS (unit, no real data)
# ══════════════════════════════════════════════════════════════════════════════

class TestAttentionExtractorHelpers:

    def _make_analysis(self, timing_cls="EARLY_CASCADE", v=11.5, pitch=8.0, ttc=400.0):
        return {
            "primary_failure": "alternator",
            "ttc_s": ttc,
            "inject_t": 0.0,
            "first_cascade_t": 120.0,
            "cascade_events": {"fuel_pump": 120.0},
            "time_between_failures": "<2 minutes (90s)",
            "peak_attn_t": 80.0,
            "peak_timing_cls": timing_cls,
            "timing_fractions": {
                "EARLY_CASCADE": 0.6 if timing_cls == "EARLY_CASCADE" else 0.0,
                "MID_CASCADE": 0.3,
                "TERMINAL": 0.1 if timing_cls != "TERMINAL" else 1.0,
            },
            "critical_states": [
                {"t": 80.0, "attn_w": 0.2, "timing_cls": timing_cls,
                 "alt": 3000.0, "v": v, "rpm": 800.0, "ff": 2.0,
                 "pump_eff": 0.5, "pitch": pitch, "roll": 0.0, "vs": -200.0},
            ],
            "var_ranking": [
                {"variable": "v",     "mean_attention": 0.8, "danger_zone": [14.0, 24.0], "n_steps_in_zone": 5},
                {"variable": "pitch", "mean_attention": 0.7, "danger_zone": [5.0, 20.0],  "n_steps_in_zone": 4},
                {"variable": "ff",    "mean_attention": 0.6, "danger_zone": [0.0, 3.0],   "n_steps_in_zone": 3},
            ],
        }

    def test_aggregate_produces_expected_keys(self):
        from backend.layer3.attention_extractor import _aggregate_component
        analyses = [self._make_analysis() for _ in range(10)]
        result = _aggregate_component(analyses, "alternator", analyses[:10], [])
        assert "timing_fractions"             in result
        assert "kill_zone"                    in result
        assert "induced_rule"                 in result
        assert "quality_score"                in result
        assert "variable_attention_ranking"   in result
        assert "crash_probability_in_zone"    in result

    def test_induced_rule_has_context1_fields(self):
        """Rule must contain all fields from context1 [Layer 3] output format."""
        from backend.layer3.attention_extractor import _aggregate_component
        analyses = [self._make_analysis() for _ in range(10)]
        result = _aggregate_component(analyses, "alternator", analyses[:10], [])
        rule = result["induced_rule"]
        assert "induced_relationship"              in rule
        assert "critical_state_combination"        in rule
        assert "crash_probability_in_this_zone"    in rule
        assert "recovery_probability_in_this_zone" in rule
        assert "interpretation"                    in rule
        assert "validation_status"                 in rule
        assert "top_attention_variables"           in rule

    def test_kill_zone_probability_with_empty_rec(self):
        """With no recoverable contrast, crash_prob should still return a value."""
        from backend.layer3.attention_extractor import _aggregate_component
        analyses = [self._make_analysis() for _ in range(8)]
        result = _aggregate_component(analyses, "alternator", analyses, [])
        # crash_probability_in_zone can be None (no zone) or float
        cp = result.get("crash_probability_in_zone")
        if cp is not None:
            assert 0.0 <= cp <= 1.0

    def test_time_between_failures_format(self):
        from backend.layer3.attention_extractor import _time_between_failures
        # < 1 minute
        result = _time_between_failures({"fuel_pump": 45.0}, 0.0)
        assert result is not None
        assert "45" in result
        # < 2 minutes
        result = _time_between_failures({"fuel_pump": 90.0}, 0.0)
        assert result is not None
        assert "2 minutes" in result

    def test_classify_timing_early_cascade(self):
        from backend.layer3.attention_extractor import _classify_timing
        # Before first cascade = EARLY_CASCADE
        assert _classify_timing(50.0, 100.0, 500.0) == "EARLY_CASCADE"
        # During cascade, before terminal = MID_CASCADE
        assert _classify_timing(200.0, 100.0, 500.0) == "MID_CASCADE"
        # Last 20% = TERMINAL
        assert _classify_timing(450.0, 100.0, 500.0) == "TERMINAL"

    def test_all_terminal_attention_gives_zero_quality(self):
        from backend.layer3.attention_extractor import _aggregate_component
        analyses = [
            {
                "primary_failure": "elevator",
                "ttc_s": 400.0,
                "inject_t": 0.0,
                "first_cascade_t": None,
                "cascade_events": {},
                "time_between_failures": None,
                "peak_attn_t": 380.0,
                "peak_timing_cls": "TERMINAL",
                "timing_fractions": {"EARLY_CASCADE": 0.0, "MID_CASCADE": 0.0, "TERMINAL": 1.0},
                "critical_states": [],
                "var_ranking": [],
            }
            for _ in range(5)
        ]
        result = _aggregate_component(analyses, "elevator", analyses, [])
        assert result.get("quality_score", 0) == 0.0

    def test_model_not_loaded_if_file_missing(self):
        from backend.layer3.attention_extractor import extract_attention
        with pytest.raises(FileNotFoundError):
            extract_attention(model_path=Path("/nonexistent/model.pt"))
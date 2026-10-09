"""Configuration contracts for offline acoustic batch controls."""

from unittest.mock import Mock

import pytest

from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.sources import (
    patches_from_dotted_cli,
    patches_from_stages_mapping,
)
from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.config import (
    NemotronVoiceChatPipelineConfig,
    PerceptionFactoryArgs,
    default_perception_factory_args,
)


@pytest.mark.parametrize(
    ("controls", "valid"),
    [
        ({"max_batch_size": 4, "max_batch_wait_ms": 1.5}, True),
        ({"max_cached_batch_runners": 1}, True),
        ({"max_cached_batch_runners": 3}, True),
        ({"max_cached_batch_runners": True}, False),
        ({"max_cached_batch_runners": False}, False),
        ({"max_cached_batch_runners": 1.5}, False),
        ({"max_cached_batch_runners": 0}, False),
        ({"max_cached_batch_runners": -1}, False),
        ({"max_batch_size": True}, False),
        ({"max_batch_size": 1.5}, False),
        ({"max_batch_frames": False}, False),
        ({"max_batch_frames": 0}, False),
        ({"max_batch_wait_ms": float("nan")}, False),
        ({"max_padding_ratio": float("inf")}, False),
        ({"max_padding_ratio": 0.99}, False),
        ({"enable_cuda_graph": 1}, False),
    ],
)
def test_factory_rejects_invalid_controls_before_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
    controls: dict[str, int | float | bool],
    valid: bool,
) -> None:
    loader = Mock(side_effect=RuntimeError("reached weight loading"))
    monkeypatch.setattr(stages, "perception_config", loader)
    if valid:
        with pytest.raises(RuntimeError, match="reached weight loading"):
            stages.create_perception_executor("test", device="cpu", **controls)
        assert loader.call_count == 1
    else:
        with pytest.raises(ValueError):
            stages.create_perception_executor("test", device="cpu", **controls)
        assert loader.call_count == 0


def test_model_config_yaml_cli_and_replacement_preserve_controls() -> None:
    config = NemotronVoiceChatPipelineConfig(model_path="test")
    cli = patches_from_dotted_cli(
        [
            ("perception.factory.max_batch_size", "8"),
            ("perception.factory.max_cached_batch_runners", "3"),
            ("perception.factory.max_batch_frames", "512"),
            ("perception.factory.max_padding_ratio", "1.5"),
            ("perception.factory.max_batch_wait_ms", "0.25"),
        ],
        config,
    )
    resolved = ConfigResolver(config).resolve(cli).config
    roundtrip = NemotronVoiceChatPipelineConfig.model_validate(resolved.model_dump())
    factory = roundtrip.stage_named("perception").factory
    assert factory.max_batch_size == 8
    assert factory.max_cached_batch_runners == 3
    assert factory.max_batch_frames == 512
    assert factory.max_padding_ratio == 1.5
    assert factory.max_batch_wait_ms == 0.25
    yaml = patches_from_stages_mapping(
        {
            "perception": {
                "factory": {
                    "max_batch_size": 4,
                    "max_batch_wait_ms": 0.75,
                    "max_cached_batch_runners": 1,
                }
            }
        },
        type(config),
        ["perception"],
    )
    resolved_yaml = ConfigResolver(config).resolve(yaml).config
    assert resolved_yaml.stage_named("perception").factory.max_batch_size == 4
    assert resolved_yaml.stage_named("perception").factory.max_cached_batch_runners == 1
    assert resolved_yaml.stage_named("perception").factory.max_batch_wait_ms == 0.75


@pytest.mark.parametrize("control_name", ["max_batch_size", "max_cached_batch_runners"])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, 0, -1])
def test_yaml_keeps_strict_integer_controls(
    control_name: str, bad_value: int | float | bool
) -> None:
    config = NemotronVoiceChatPipelineConfig(model_path="test")
    with pytest.raises(ValueError):
        patch = patches_from_stages_mapping(
            {"perception": {"factory": {control_name: bad_value}}},
            type(config),
            ["perception"],
        )
        ConfigResolver(config).resolve(patch)


def test_cache_default_has_one_config_owner() -> None:
    assert PerceptionFactoryArgs().max_cached_batch_runners is None
    assert default_perception_factory_args().max_cached_batch_runners == 1
    config = NemotronVoiceChatPipelineConfig(model_path="test")
    assert config.stage_named("perception").factory.max_cached_batch_runners == 1
    assert config.stage_named("perception").factory.max_batch_size == 1


@pytest.mark.parametrize("bad_value", [True, False, 1.5, 0, -1])
def test_raw_cache_config_rejects_invalid_values(bad_value: int | float | bool) -> None:
    with pytest.raises(ValueError):
        PerceptionFactoryArgs(max_cached_batch_runners=bad_value)


@pytest.mark.parametrize("bad_value", ["true", "false", "1.5", "0", "-1"])
def test_cli_cache_override_rejects_invalid_values(bad_value: str) -> None:
    config = NemotronVoiceChatPipelineConfig(model_path="test")
    with pytest.raises(ValueError):
        patches = patches_from_dotted_cli(
            [("perception.factory.max_cached_batch_runners", bad_value)],
            config,
        )
        ConfigResolver(config).resolve(patches)

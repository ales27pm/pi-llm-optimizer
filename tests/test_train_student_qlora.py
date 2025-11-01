import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict

import pytest

from automation.pipeline_ops import TrainStudentConfig
from desktop_distill import train_student


BF16_PRESETS = sorted(
    name
    for name, preset in train_student.QLORA_PRESETS.items()
    if preset.compute_dtype == "bfloat16"
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "<|eos|>"

    def __call__(self, texts, truncation: bool, max_length: int, padding: bool) -> dict[str, Any]:
        lengths = [max(1, min(len(text), max_length)) for text in texts]
        return {
            "input_ids": [[0] * length for length in lengths],
            "attention_mask": [[1] * length for length in lengths],
        }

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyDataset:
    def __init__(self, dataset_path: Path) -> None:
        with dataset_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise AssertionError("synthetic dataset must not be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class DummyModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"use_cache": True})()
        self.gradient_checkpoint_enabled = False
        self.saved_paths: list[Path] = []
        self.trainable_parameters_logged = False
        self.prepared_for_kbit = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpoint_enabled = True

    def merge_and_unload(self) -> "DummyModel":
        return self

    def to(self, target):
        # Allow chaining for both device and dtype transitions
        return self

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.bin").write_bytes(b"dummy")

    def print_trainable_parameters(self) -> None:
        self.trainable_parameters_logged = True


class DummyTrainer:
    last_instance: "DummyTrainer | None" = None

    def __init__(self, *, model, args, train_dataset, data_collator) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.train_called = False
        DummyTrainer.last_instance = self

    def train(self) -> None:
        self.train_called = True


class DummyTrainingArguments(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_student,
        "_prepare_tokenizer",
        lambda base_model: DummyTokenizer(),
    )


def _patch_data_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_student,
        "_build_data_pipeline",
        lambda dataset, tokenizer, max_length: (
            DummyDataset(Path(dataset)),
            lambda batch: batch,
        ),
    )


def _patch_model_loader(
    monkeypatch: pytest.MonkeyPatch,
    dummy_model: "DummyModel",
    quant_snapshot: Dict[str, Any],
) -> None:
    def _fake_from_pretrained(base_model, torch_dtype=None, **kwargs):
        quant_config = kwargs.get("quantization_config")
        assert (
            quant_config is not None
        ), "QLoRA preset must provide quantization_config"
        quant_snapshot["load_in_4bit"] = getattr(quant_config, "load_in_4bit", False)
        quant_snapshot["quant_type"] = quant_config.bnb_4bit_quant_type
        quant_snapshot["double_quant"] = quant_config.bnb_4bit_use_double_quant
        quant_snapshot["compute_dtype"] = quant_config.bnb_4bit_compute_dtype
        return dummy_model

    monkeypatch.setattr(
        train_student.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )


def _patch_prepare_for_kbit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _prepare_for_kbit(model, use_gradient_checkpointing=True):
        model.prepared_for_kbit = True
        return model

    monkeypatch.setattr(train_student, "prepare_model_for_kbit_training", _prepare_for_kbit)


def _patch_adapter_capture(
    monkeypatch: pytest.MonkeyPatch,
    captured_adapter: Dict[str, Any],
) -> None:
    def _capture_adapter(model, adapter_config):
        captured_adapter["rank"] = adapter_config.r
        captured_adapter["alpha"] = adapter_config.lora_alpha
        captured_adapter["dropout"] = adapter_config.lora_dropout
        captured_adapter["targets"] = tuple(adapter_config.target_modules)
        return model

    monkeypatch.setattr(train_student, "get_peft_model", _capture_adapter)


def _patch_trainer_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_student, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_student, "TrainingArguments", DummyTrainingArguments)


def _patch_cuda_support(
    monkeypatch: pytest.MonkeyPatch, *, bf16_supported: bool
) -> None:
    monkeypatch.setattr(train_student.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        train_student.torch.cuda,
        "is_bf16_supported",
        lambda: bf16_supported,
    )


def _patch_arg_parser(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_path: Path,
    output_dir: Path,
    preset_name: str,
) -> None:
    monkeypatch.setattr(
        train_student,
        "_parse_args",
        lambda: argparse.Namespace(
            dataset=dataset_path,
            base_model="dummy/base",
            output_dir=output_dir,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            use_dora=False,
            qlora=True,
            qlora_preset=preset_name,
            seed=None,
            logging_steps=None,
            max_length=1024,
            lr_scheduler_type="cosine",
            warmup_steps=0,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="q_proj,k_proj,v_proj,o_proj",
        ),
    )


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    data_path = tmp_path / "synthetic.jsonl"
    sample = {
        "system_hint": "You are a concise assistant.",
        "user": "Ping?",
        "assistant": "Pong!",
    }
    data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return data_path


@pytest.fixture()
def qlora_training_run(
    monkeypatch: pytest.MonkeyPatch, synthetic_dataset: Path, tmp_path: Path
) -> Callable[..., Dict[str, Any]]:
    """Return a callable that executes ``train_student.main`` for a preset."""

    def _runner(preset_name: str, *, bf16_supported: bool = True) -> Dict[str, Any]:
        output_dir = tmp_path / f"model-{preset_name}"
        dummy_model = DummyModel()
        captured_adapter: Dict[str, Any] = {}
        quant_snapshot: Dict[str, Any] = {}

        _patch_tokenizer(monkeypatch)
        _patch_data_pipeline(monkeypatch)
        _patch_model_loader(monkeypatch, dummy_model, quant_snapshot)
        _patch_prepare_for_kbit(monkeypatch)
        _patch_adapter_capture(monkeypatch, captured_adapter)
        _patch_trainer_components(monkeypatch)
        _patch_cuda_support(monkeypatch, bf16_supported=bf16_supported)
        _patch_arg_parser(
            monkeypatch,
            dataset_path=synthetic_dataset,
            output_dir=output_dir,
            preset_name=preset_name,
        )

        DummyTrainer.last_instance = None

        train_student.main()

        assert DummyTrainer.last_instance is not None

        return {
            "adapter": captured_adapter,
            "quant_config": quant_snapshot,
            "model": dummy_model,
            "trainer": DummyTrainer.last_instance,
            "output_dir": output_dir,
        }

    return _runner


@pytest.mark.parametrize("preset_name", sorted(train_student.QLORA_PRESETS.keys()))
def test_qlora_presets_cover_matrix(qlora_training_run, preset_name: str) -> None:
    result = qlora_training_run(preset_name)

    preset = train_student.QLORA_PRESETS[preset_name]
    quant = result["quant_config"]
    adapter = result["adapter"]

    assert quant["load_in_4bit"] is True
    assert quant["quant_type"] == preset.quant_type
    assert quant["double_quant"] is preset.double_quant

    dtype_map = {
        "bfloat16": train_student.torch.bfloat16,
        "float16": train_student.torch.float16,
    }
    assert quant["compute_dtype"] == dtype_map[preset.compute_dtype]

    assert adapter["rank"] == preset.lora_rank
    assert adapter["alpha"] == pytest.approx(preset.lora_alpha)
    assert adapter["dropout"] == pytest.approx(preset.lora_dropout)

    expected_targets = tuple(
        module.strip()
        for module in (preset.target_modules or "q_proj,k_proj,v_proj,o_proj").split(",")
        if module.strip()
    )
    assert set(adapter["targets"]) == set(expected_targets)

    dummy_model: DummyModel = result["model"]
    trainer: DummyTrainer = result["trainer"]
    output_dir: Path = result["output_dir"]

    assert dummy_model.gradient_checkpoint_enabled is True
    assert dummy_model.trainable_parameters_logged is True
    assert dummy_model.prepared_for_kbit is True
    assert trainer.train_called is True
    adapter_path = output_dir / "adapter.bin"
    tokenizer_path = output_dir / "tokenizer.json"
    assert adapter_path.exists()
    assert tokenizer_path.exists()
    assert adapter_path.stat().st_size > 0
    assert tokenizer_path.stat().st_size > 0


@pytest.mark.parametrize("preset_name", BF16_PRESETS)
def test_qlora_preset_warns_when_bf16_missing(
    qlora_training_run, preset_name: str
) -> None:
    with pytest.warns(RuntimeWarning):
        result = qlora_training_run(preset_name, bf16_supported=False)

    quant = result["quant_config"]
    assert quant["compute_dtype"] == train_student.torch.float16


def test_pipeline_command_includes_preset(tmp_path: Path) -> None:
    config = TrainStudentConfig(
        dataset=tmp_path / "data.jsonl",
        base_model="dummy/base",
        output_dir=tmp_path / "out",
        qlora=False,
        qlora_preset="turing-safe",
    )
    command = config.build_command(python_executable="python")
    assert "--qlora" in command
    assert "--qlora_preset" in command
    preset_index = command.index("--qlora_preset")
    assert command[preset_index + 1] == "turing-safe"


def test_pipeline_unknown_preset_rejected(tmp_path: Path) -> None:
    config = TrainStudentConfig(
        dataset=tmp_path / "data.jsonl",
        base_model="dummy/base",
        output_dir=tmp_path / "out",
        qlora_preset="not-real",
    )
    with pytest.raises(ValueError):
        config.build_command(python_executable="python")

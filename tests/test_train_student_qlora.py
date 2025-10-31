import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from automation.pipeline_ops import TrainStudentConfig
from desktop_distill import train_student


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


def test_qlora_preset_execution(monkeypatch: pytest.MonkeyPatch, synthetic_dataset: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "model"
    dummy_model = DummyModel()
    captured_adapter: dict[str, Any] = {}

    monkeypatch.setattr(train_student, "_prepare_tokenizer", lambda base_model: DummyTokenizer())
    monkeypatch.setattr(train_student, "_build_data_pipeline", lambda dataset, tokenizer, max_length: (DummyDataset(dataset), lambda batch: batch))

    def _assert_quant_config(kwargs: dict[str, Any]) -> None:
        quant_config = kwargs.get("quantization_config")
        assert quant_config is not None, "QLoRA preset must provide quantization_config"
        assert quant_config.bnb_4bit_quant_type == "nf4"
        assert quant_config.bnb_4bit_use_double_quant is True
        assert quant_config.bnb_4bit_compute_dtype == train_student.torch.bfloat16

    def _fake_from_pretrained(base_model, torch_dtype=None, **kwargs):
        _assert_quant_config(kwargs)
        return dummy_model

    monkeypatch.setattr(
        train_student.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )

    monkeypatch.setattr(train_student, "prepare_model_for_kbit_training", lambda model, use_gradient_checkpointing=True: model)

    def _capture_adapter(model, adapter_config):
        captured_adapter["rank"] = adapter_config.r
        captured_adapter["alpha"] = adapter_config.lora_alpha
        captured_adapter["dropout"] = adapter_config.lora_dropout
        captured_adapter["targets"] = tuple(adapter_config.target_modules)
        return model

    monkeypatch.setattr(train_student, "get_peft_model", _capture_adapter)
    monkeypatch.setattr(train_student, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_student, "TrainingArguments", DummyTrainingArguments)

    monkeypatch.setattr(train_student.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_student.torch.cuda, "is_bf16_supported", lambda: True)

    monkeypatch.setattr(
        train_student,
        "_parse_args",
        lambda: argparse.Namespace(
            dataset=synthetic_dataset,
            base_model="dummy/base",
            output_dir=output_dir,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            use_dora=False,
            qlora=True,
            qlora_preset="ampere-balanced",
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

    train_student.main()

    assert DummyTrainer.last_instance is not None
    assert DummyTrainer.last_instance.train_called is True
    assert dummy_model.trainable_parameters_logged is True
    assert captured_adapter["rank"] == 64
    assert captured_adapter["alpha"] == 128
    assert captured_adapter["dropout"] == 0.05
    assert set(captured_adapter["targets"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert (output_dir / "adapter.bin").exists()
    assert (output_dir / "tokenizer.json").exists()


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

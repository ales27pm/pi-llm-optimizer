from __future__ import annotations

from pathlib import Path

import pytest

from automation.pipeline_ops import (
    BenchmarkConfig,
    ExportConfig,
    TeacherLabelConfig,
    TrainStudentConfig,
)


def test_teacher_label_builds_expected_command(tmp_path: Path) -> None:
    config = TeacherLabelConfig(
        model="openai/dolphin",
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
        max_new_tokens=128,
        do_sample=True,
        temperature=0.9,
        device_map="auto",
        batch_size=2,
        skip_existing=True,
    )

    command = config.build_command(python_executable="python")

    assert command[:2] == ["python", str(Path("desktop_distill/teacher_label.py").resolve())]
    assert "--model" in command and command[command.index("--model") + 1].endswith("openai/dolphin")
    assert "--do_sample" in command
    assert "--skip_existing" in command


def test_train_student_validates_batch_size() -> None:
    config = TrainStudentConfig(
        dataset=Path("dataset.jsonl"),
        base_model="test/base",
        output_dir=Path("out"),
        batch_size=0,
    )
    with pytest.raises(ValueError):
        config.build_command(python_executable="python")


def test_benchmark_command_optional_fields(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        model=tmp_path / "model.gguf",
        iterations=5,
        min_tokps=0.42,
        csv_path=tmp_path / "results.csv",
        json_path=tmp_path / "history.json",
        json_limit=10,
    )

    command = config.build_command(python_executable="python")

    assert "--min-tokps" in command
    assert "--csv" in command
    assert "--json" in command
    assert "--json-limit" in command


def test_export_includes_token_when_provided(tmp_path: Path) -> None:
    config = ExportConfig(
        model_path=tmp_path,
        output_dir=tmp_path,
        quant_type="q5_k_m",
        hf_token="secret",
    )

    command = config.build_command(python_executable="python")

    assert command.count("--hf-token") == 1
    assert command[command.index("--qtype") + 1] == "q5_k_m"

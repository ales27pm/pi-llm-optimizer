"""Interactive Textual UI for orchestrating the Pi-LLM optimisation workflow."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Tab,
    Tabs,
    TextLog,
)

from .pipeline_ops import BenchmarkConfig, ExportConfig, TeacherLabelConfig, TrainStudentConfig

APP_CSS_PATH = Path(__file__).with_name("ui_app.tcss")


class PipelineApp(App[None]):
    """A multi-panel Textual application for running pipeline stages."""

    CSS_PATH = APP_CSS_PATH
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("f1", "switch('label')", "Teacher labelling"),
        ("f2", "switch('train')", "Student training"),
        ("f3", "switch('export')", "GGUF export"),
        ("f4", "switch('bench')", "Benchmark"),
        ("f10", "clear_log", "Clear log"),
    ]

    current_panel: reactive[str] = reactive("label")

    def compose(self) -> ComposeResult:  # pragma: no cover - UI composition
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="sidebar"):
                yield Label("Pipeline stages", id="sidebar-title")
                yield Tabs(
                    Tab("Teacher labelling", id="label"),
                    Tab("Student training", id="train"),
                    Tab("GGUF export", id="export"),
                    Tab("Benchmark", id="bench"),
                    id="nav-tabs",
                )
                with Container(id="panel-holder"):
                    yield self._build_teacher_panel()
                    yield self._build_train_panel()
                    yield self._build_export_panel()
                    yield self._build_benchmark_panel()
            yield TextLog(id="log", highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime behaviour
        self._switch_panel("label")

    async def action_switch(self, panel_id: str) -> None:  # pragma: no cover - runtime behaviour
        self._switch_panel(panel_id)

    async def action_clear_log(self) -> None:  # pragma: no cover - runtime behaviour
        log = self.query_one(TextLog)
        log.clear()

    @on(Tabs.TabActivated)
    def _when_tab_selected(self, event: Tabs.TabActivated) -> None:  # pragma: no cover - UI event
        self._switch_panel(event.tab.id)

    def _switch_panel(self, panel_id: str) -> None:
        panel_holder = self.query_one("#panel-holder")
        for child in panel_holder.children:
            child.display = child.id == f"panel-{panel_id}"
        self.current_panel = panel_id

    # region Panel builders -------------------------------------------------

    def _build_teacher_panel(self) -> Container:
        return self._form_panel(
            panel_id="label",
            title="Teacher labelling",
            description=(
                "Generate assistant completions for your dataset using a large teacher model."
            ),
            controls=[
                Input(placeholder="HF model id", id="label-model"),
                Input(placeholder="Input dataset path", id="label-input"),
                Input(placeholder="Output dataset path", id="label-output"),
                Input(placeholder="Max new tokens (default 256)", id="label-max-new-tokens"),
                Checkbox(label="Enable sampling", id="label-do-sample"),
                Input(placeholder="Temperature (default 0.7)", id="label-temperature"),
                Input(placeholder="Device map (optional)", id="label-device-map"),
                Input(placeholder="Batch size (default 1)", id="label-batch-size"),
                Checkbox(label="Skip already labelled records", id="label-skip-existing"),
                Button("Run labelling", id="run-label"),
            ],
        )

    def _build_train_panel(self) -> Container:
        return self._form_panel(
            panel_id="train",
            title="Student training",
            description="Fine tune the distilled student model with LoRA/DoRA adapters.",
            controls=[
                Input(placeholder="Labelled dataset path", id="train-dataset"),
                Input(placeholder="Base model (HF id)", id="train-base-model"),
                Input(placeholder="Output directory", id="train-output-dir"),
                Input(placeholder="Epochs (default 1)", id="train-num-epochs"),
                Input(placeholder="Batch size (default 1)", id="train-batch-size"),
                Input(placeholder="Gradient accumulation steps (default 1)", id="train-grad-steps"),
                Input(placeholder="Learning rate (default 2e-5)", id="train-learning-rate"),
                Checkbox(label="Use DoRA adapters", id="train-use-dora"),
                Checkbox(label="Enable QLoRA", id="train-qlora"),
                Input(placeholder="Seed (optional)", id="train-seed"),
                Input(placeholder="Logging steps (optional)", id="train-logging-steps"),
                Button("Run training", id="run-train"),
            ],
        )

    def _build_export_panel(self) -> Container:
        return self._form_panel(
            panel_id="export",
            title="GGUF export",
            description="Convert the fine-tuned model into GGUF format and quantise it.",
            controls=[
                Input(placeholder="Trained model directory", id="export-model"),
                Input(placeholder="Output directory", id="export-outdir"),
                Input(placeholder="Quantisation type (default q4_k_m)", id="export-qtype"),
                Input(placeholder="HF token (optional)", password=True, id="export-hf-token"),
                Button("Run export", id="run-export"),
            ],
        )

    def _build_benchmark_panel(self) -> Container:
        return self._form_panel(
            panel_id="bench",
            title="Benchmark",
            description="Measure throughput and latency on the Raspberry Pi runtime.",
            controls=[
                Input(placeholder="GGUF model path", id="bench-model"),
                Input(placeholder="Iterations (default 3)", id="bench-iterations"),
                Input(placeholder="Minimum tok/s (optional)", id="bench-min-tokps"),
                Input(placeholder="CSV output path (optional)", id="bench-csv"),
                Button("Run benchmark", id="run-bench"),
            ],
        )

    def _form_panel(
        self,
        *,
        panel_id: str,
        title: str,
        description: str,
        controls: list[object],
    ) -> Container:
        panel = Container(
            Vertical(
                Label(title, classes="panel-title"),
                Label(description, classes="panel-description"),
                *controls,
                classes="panel-body",
            ),
            id=f"panel-{panel_id}",
        )
        panel.display = False
        return panel

    # endregion

    # region Event handlers -------------------------------------------------

    @on(Button.Pressed, "#run-label")
    async def _run_labelling(self) -> None:  # pragma: no cover - runtime behaviour
        try:
            config = TeacherLabelConfig(
                model=self._require_value("#label-model"),
                input_path=Path(self._require_value("#label-input")),
                output_path=Path(self._require_value("#label-output")),
                max_new_tokens=self._optional_int("#label-max-new-tokens", default=256),
                do_sample=self.query_one("#label-do-sample", Checkbox).value,
                temperature=self._optional_float("#label-temperature", default=0.7),
                device_map=self._optional_str("#label-device-map"),
                batch_size=self._optional_int("#label-batch-size", default=1),
                skip_existing=self.query_one("#label-skip-existing", Checkbox).value,
            )
        except ValueError as exc:
            self._report_error(str(exc))
            return
        try:
            command = config.build_command()
        except ValueError as exc:
            self._report_error(str(exc))
            return
        await self._execute(command, "Teacher labelling")

    @on(Button.Pressed, "#run-train")
    async def _run_training(self) -> None:  # pragma: no cover - runtime behaviour
        try:
            config = TrainStudentConfig(
                dataset=Path(self._require_value("#train-dataset")),
                base_model=self._require_value("#train-base-model"),
                output_dir=Path(self._require_value("#train-output-dir")),
                num_epochs=self._optional_int("#train-num-epochs", default=1),
                batch_size=self._optional_int("#train-batch-size", default=1),
                gradient_accumulation_steps=self._optional_int("#train-grad-steps", default=1),
                learning_rate=self._optional_float("#train-learning-rate", default=2e-5),
                use_dora=self.query_one("#train-use-dora", Checkbox).value,
                qlora=self.query_one("#train-qlora", Checkbox).value,
                seed=self._optional_int("#train-seed"),
                logging_steps=self._optional_int("#train-logging-steps"),
            )
        except ValueError as exc:
            self._report_error(str(exc))
            return
        try:
            command = config.build_command()
        except ValueError as exc:
            self._report_error(str(exc))
            return
        await self._execute(command, "Student training")

    @on(Button.Pressed, "#run-export")
    async def _run_export(self) -> None:  # pragma: no cover - runtime behaviour
        try:
            config = ExportConfig(
                model_path=Path(self._require_value("#export-model")),
                output_dir=Path(self._require_value("#export-outdir")),
                quant_type=self._optional_str("#export-qtype", default="q4_k_m"),
                hf_token=self._optional_str("#export-hf-token"),
            )
        except ValueError as exc:
            self._report_error(str(exc))
            return
        try:
            command = config.build_command()
        except ValueError as exc:
            self._report_error(str(exc))
            return
        await self._execute(command, "GGUF export")

    @on(Button.Pressed, "#run-bench")
    async def _run_benchmark(self) -> None:  # pragma: no cover - runtime behaviour
        try:
            config = BenchmarkConfig(
                model=Path(self._require_value("#bench-model")),
                iterations=self._optional_int("#bench-iterations", default=3),
                min_tokps=self._optional_float("#bench-min-tokps"),
                csv_path=self._optional_path("#bench-csv"),
            )
        except ValueError as exc:
            self._report_error(str(exc))
            return
        try:
            command = config.build_command()
        except ValueError as exc:
            self._report_error(str(exc))
            return
        await self._execute(command, "Benchmark")

    # endregion

    # region Helpers --------------------------------------------------------

    def _require_value(self, selector: str) -> str:
        widget = self.query_one(selector, Input)
        value = widget.value.strip()
        if not value:
            raise ValueError(f"Field '{selector}' is required")
        return value

    def _optional_str(self, selector: str, default: Optional[str] = None) -> Optional[str]:
        widget = self.query_one(selector, Input)
        value = widget.value.strip()
        return value or default

    def _optional_int(self, selector: str, default: Optional[int] = None) -> Optional[int]:
        value = self._optional_str(selector)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - validated via runtime usage
            raise ValueError(f"Field '{selector}' must be an integer") from exc

    def _optional_float(self, selector: str, default: Optional[float] = None) -> Optional[float]:
        value = self._optional_str(selector)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - validated via runtime usage
            raise ValueError(f"Field '{selector}' must be a number") from exc

    def _optional_path(self, selector: str) -> Optional[Path]:
        value = self._optional_str(selector)
        return Path(value) if value else None

    def _report_error(self, message: str) -> None:
        log = self.query_one(TextLog)
        log.write(f"[bold red]Input error:[/bold red] {message}")

    async def _execute(self, command: list[str], label: str) -> None:
        log = self.query_one(TextLog)
        log.write(f"[bold green]▶ Starting {label}[/bold green]")
        log.write("[dim]$ %s[/dim]" % " ".join(command))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:  # pragma: no cover - runtime guard
            log.write(f"[bold red]Failed to start process:[/bold red] {exc}")
            return

        async def _stream(stream: asyncio.StreamReader, prefix: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                log.write(f"{prefix}{line.decode().rstrip()}")

        await asyncio.gather(
            _stream(process.stdout, ""),
            _stream(process.stderr, "[red]⚠[/red] "),
        )
        return_code = await process.wait()
        if return_code == 0:
            log.write(f"[bold green]✔ {label} completed successfully[/bold green]")
        else:
            log.write(f"[bold red]✖ {label} failed with exit code {return_code}[/bold red]")


if __name__ == "__main__":  # pragma: no cover
    PipelineApp().run()

"""Interactive Textual UI for orchestrating the Pi-LLM optimisation workflow."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Sparkline,
    Tab,
    Tabs,
    TextLog,
)

from .pipeline_ops import (
    BenchmarkConfig,
    ExportConfig,
    PROTOCOL_METADATA,
    TeacherLabelConfig,
    TrainStudentConfig,
    QLORA_PRESETS,
)

from rpi4.bench.history_utils import load_history

if TYPE_CHECKING:
    from desktop_distill.train_student import QLoRAPreset

APP_CSS_PATH = Path(__file__).with_name("ui_app.tcss")
DEFAULT_BENCHMARK_HISTORY = Path("rpi4/bench/out/bench_history.json")


QLORA_PRESET_OPTIONS = [
    (preset.label, name)
    for name, preset in sorted(QLORA_PRESETS.items(), key=lambda item: item[0])
]


DEFAULT_PRESET_DETAILS = "Select a preset to populate quantisation and LoRA hyperparameters."
MISSING_PRESET_DETAILS = "Preset metadata missing; verify desktop_distill/train_student.py."
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"


class BenchmarkDashboard(Vertical):
    """Render benchmark history with trend visualisations."""

    def __init__(
        self,
        history_path: Path,
        refresh_interval: float,
        *,
        max_entries: int = 20,
    ) -> None:
        super().__init__(id="benchmark-dashboard")
        self._history_path = history_path
        self._refresh_interval = refresh_interval
        self._max_entries = max_entries
        self._status = Label("Awaiting benchmark results…", id="bench-dashboard-status")
        self._sparkline = Sparkline(id="bench-dashboard-sparkline")
        self._table = DataTable(id="bench-dashboard-table", zebra_stripes=True)

    def compose(self) -> ComposeResult:  # pragma: no cover - UI composition
        yield Label("Benchmark trends", classes="panel-subtitle")
        yield self._status
        yield self._sparkline
        yield self._table

    def on_mount(self) -> None:  # pragma: no cover - runtime behaviour
        self._table.add_columns("Timestamp", "Avg tok/s", "Min tok/s", "Model")
        self.set_interval(self._refresh_interval, self.refresh_data, pause=False)
        self.refresh_data()

    def refresh_data(self) -> None:  # pragma: no cover - runtime behaviour
        runs = load_history(self._history_path)
        if not runs:
            self._status.update(f"No history found at {self._history_path}")
            self._sparkline.data = []
            self._table.clear()
            return

        latest_runs = runs[-self._max_entries:]
        averages = [
            value
            for value in (_coerce_float(run, "summary", "average_tokps") for run in latest_runs)
            if value is not None
        ]
        self._table.clear()
        for run in reversed(latest_runs):
            avg = _coerce_float(run, "summary", "average_tokps")
            minimum = _coerce_float(run, "summary", "minimum_observed_tokps")
            timestamp = self._format_timestamp(run.get("timestamp"))
            model_name = Path(str(run.get("model", "?"))).name
            self._table.add_row(
                timestamp,
                f"{avg:.2f}" if avg is not None else "-",
                f"{minimum:.2f}" if minimum is not None else "-",
                model_name,
            )

        if averages:
            self._sparkline.data = averages
        else:
            self._sparkline.data = []

        last_run = latest_runs[-1]
        last_timestamp = self._format_timestamp(last_run.get("timestamp"))
        last_avg = _coerce_float(last_run, "summary", "average_tokps")
        if last_avg is not None:
            self._status.update(
                f"Last run {last_timestamp} • avg {last_avg:.2f} tok/s"
            )
        else:
            self._status.update(f"Last run {last_timestamp}")

    @staticmethod
    def _format_timestamp(value: Optional[str]) -> str:
        if not value:
            return "unknown"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _coerce_float(payload: dict[str, Any], *keys: str) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


class QLoRaPresetController:
    """Encapsulate QLoRA preset selection side effects for the UI."""

    def __init__(self, app: "PipelineApp") -> None:
        self._checkbox = app.query_one("#train-qlora", Checkbox)
        self._log = app.query_one(TextLog)
        self._presets = QLORA_PRESETS
        self._details_label = app.query_one("#train-qlora-preset-details", Label)
        self._details_label.update(DEFAULT_PRESET_DETAILS)

    def handle(self, value: Optional[str]) -> None:
        enabled = bool(value)
        self._checkbox.value = enabled

        if not value:
            self._log.write("Cleared QLoRA preset selection.")
            self._details_label.update(DEFAULT_PRESET_DETAILS)
            return

        preset = self._presets.get(value)
        if preset is None:
            self._log.write(
                f"[yellow]WARNING:[/] Unknown preset '{value}'; using defaults."
            )
            self._details_label.update(MISSING_PRESET_DETAILS)
            return

        warning, log_message, details_message = self._build_messages(value, preset)
        if warning:
            self._log.write(warning)
        self._log.write(log_message)
        self._details_label.update(details_message)

    def _build_messages(
        self, name: str, preset: "QLoRAPreset"
    ) -> tuple[Optional[str], str, str]:
        """Return log and detail strings plus an optional warning for a preset."""

        target_modules = preset.target_modules
        warning: Optional[str] = None
        if not target_modules:
            warning = (
                f"[yellow]WARNING:[/] Preset '{name}' is missing target_modules metadata; "
                f"defaulting to {DEFAULT_TARGET_MODULES}."
            )
            target_modules = DEFAULT_TARGET_MODULES

        double_quant = "on" if preset.double_quant else "off"
        log_message = (
            f"Selected QLoRA preset '{name}' → {preset.target_gpus} • "
            f"quant={preset.quant_type}/{preset.compute_dtype} • "
            f"double-quant {double_quant} • "
            f"LoRA r={preset.lora_rank} α={preset.lora_alpha} dropout={preset.lora_dropout} • "
            f"targets={target_modules}"
        )
        details_message = " | ".join(
            [
                f"quant={preset.quant_type}/compute={preset.compute_dtype}",
                f"double-quant={double_quant}",
                f"LoRA r={preset.lora_rank} α={preset.lora_alpha} dropout={preset.lora_dropout}",
                f"targets={target_modules}",
                preset.notes or "See README for tuning notes",
            ]
        )
        return warning, log_message, details_message


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

    def __init__(
        self,
        *args,
        default_panel: str = "label",
        benchmark_history_path: Path = DEFAULT_BENCHMARK_HISTORY,
        benchmark_refresh_seconds: float = 30.0,
        dashboard_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._qlora_preset_controller: Optional[QLoRaPresetController] = None
        self._benchmark_history_path = benchmark_history_path
        self._benchmark_refresh_seconds = benchmark_refresh_seconds
        self._dashboard_only = dashboard_only
        self._available_panels = ["label", "train", "export", "bench"]
        if self._dashboard_only:
            self._available_panels = ["bench"]
        self._default_panel = default_panel if default_panel in self._available_panels else self._available_panels[0]

    def compose(self) -> ComposeResult:  # pragma: no cover - UI composition
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="sidebar"):
                yield Label("Pipeline stages", id="sidebar-title")
                tab_widgets = []
                if "label" in self._available_panels:
                    tab_widgets.append(Tab("Teacher labelling", id="label"))
                if "train" in self._available_panels:
                    tab_widgets.append(Tab("Student training", id="train"))
                if "export" in self._available_panels:
                    tab_widgets.append(Tab("GGUF export", id="export"))
                if "bench" in self._available_panels:
                    tab_widgets.append(Tab("Benchmark", id="bench"))
                yield Tabs(*tab_widgets, id="nav-tabs")
                with Container(id="panel-holder"):
                    if "label" in self._available_panels:
                        yield self._build_teacher_panel()
                    if "train" in self._available_panels:
                        yield self._build_train_panel()
                    if "export" in self._available_panels:
                        yield self._build_export_panel()
                    if "bench" in self._available_panels:
                        yield self._build_benchmark_panel()
            yield TextLog(id="log", highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime behaviour
        self._switch_panel(self._default_panel)
        log_widget = self.query_one(TextLog)
        if PROTOCOL_METADATA is not None:
            log_widget.write(
                f"Loaded agent protocol '{PROTOCOL_METADATA.title}' (version {PROTOCOL_METADATA.version})."
            )
        else:
            log_widget.write(
                "[yellow]WARNING:[/] Agent protocol metadata could not be loaded. Review AGENTS.md for format changes."
            )
        if "train" in self._available_panels:
            self._qlora_preset_controller = QLoRaPresetController(self)

    async def action_switch(self, panel_id: str) -> None:  # pragma: no cover - runtime behaviour
        self._switch_panel(panel_id)

    async def action_clear_log(self) -> None:  # pragma: no cover - runtime behaviour
        log = self.query_one(TextLog)
        log.clear()

    @on(Tabs.TabActivated)
    def _when_tab_selected(self, event: Tabs.TabActivated) -> None:  # pragma: no cover - UI event
        self._switch_panel(event.tab.id)

    @on(Select.Changed, "#train-qlora-preset")
    def _when_preset_selected(self, event: Select.Changed) -> None:  # pragma: no cover - UI event
        if self._qlora_preset_controller is None:
            self._qlora_preset_controller = QLoRaPresetController(self)
        self._qlora_preset_controller.handle(event.value if event.value else None)

    def _switch_panel(self, panel_id: str) -> None:
        if panel_id not in self._available_panels:
            panel_id = self._default_panel
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
                Select(
                    options=QLORA_PRESET_OPTIONS or [("No presets detected", "")],
                    prompt="QLoRA preset (optional)",
                    allow_blank=True,
                    id="train-qlora-preset",
                    disabled=not QLORA_PRESET_OPTIONS,
                    tooltip=(
                        "Automatically configures quantisation and LoRA hyperparameters"
                        if QLORA_PRESET_OPTIONS
                        else "Install training dependencies to load preset metadata"
                    ),
                ),
                Label(
                    "Select a preset to populate quantisation and LoRA hyperparameters.",
                    id="train-qlora-preset-details",
                    classes="preset-details",
                ),
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
                Input(placeholder="JSON history path (optional)", id="bench-json"),
                Input(placeholder="JSON history limit (default 200)", id="bench-json-limit"),
                Button("Run benchmark", id="run-bench"),
            ],
            extra_sections=[
                Label(
                    f"Reading history from {self._benchmark_history_path}",
                    classes="panel-help",
                ),
                BenchmarkDashboard(
                    self._benchmark_history_path,
                    self._benchmark_refresh_seconds,
                ),
            ],
        )

    def _form_panel(
        self,
        *,
        panel_id: str,
        title: str,
        description: str,
        controls: list[object],
        extra_sections: Optional[list[object]] = None,
    ) -> Container:
        panel = Container(
            Vertical(
                Label(title, classes="panel-title"),
                Label(description, classes="panel-description"),
                *controls,
                *(extra_sections or []),
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
            preset_choice = self._optional_select("#train-qlora-preset")
            qlora_checkbox = self.query_one("#train-qlora", Checkbox)
            qlora_enabled = qlora_checkbox.value or bool(preset_choice)
            config = TrainStudentConfig(
                dataset=Path(self._require_value("#train-dataset")),
                base_model=self._require_value("#train-base-model"),
                output_dir=Path(self._require_value("#train-output-dir")),
                num_epochs=self._optional_int("#train-num-epochs", default=1),
                batch_size=self._optional_int("#train-batch-size", default=1),
                gradient_accumulation_steps=self._optional_int("#train-grad-steps", default=1),
                learning_rate=self._optional_float("#train-learning-rate", default=2e-5),
                use_dora=self.query_one("#train-use-dora", Checkbox).value,
                qlora=qlora_enabled,
                qlora_preset=preset_choice,
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
                json_path=self._optional_path("#bench-json"),
                json_limit=self._optional_int("#bench-json-limit"),
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

    def _optional_select(self, selector: str) -> Optional[str]:
        widget = self.query_one(selector, Select)
        value = widget.value
        return None if value is Select.BLANK or value == "" else str(value)

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

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive pipeline dashboard")
    parser.add_argument(
        "--panel",
        choices=["label", "train", "export", "bench"],
        default="label",
        help="Panel to focus on at startup",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch in benchmark dashboard mode (benchmark panel only)",
    )
    parser.add_argument(
        "--benchmark-history",
        type=Path,
        default=DEFAULT_BENCHMARK_HISTORY,
        help="Path to the benchmark history JSON file",
    )
    parser.add_argument(
        "--benchmark-refresh",
        type=float,
        default=30.0,
        help="Seconds between benchmark dashboard refreshes",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper
    args = _parse_cli_args()
    refresh_interval = max(args.benchmark_refresh, 1.0)
    history_path = args.benchmark_history.expanduser().resolve()
    default_panel = "bench" if args.dashboard else args.panel
    app = PipelineApp(
        default_panel=default_panel,
        benchmark_history_path=history_path,
        benchmark_refresh_seconds=refresh_interval,
        dashboard_only=args.dashboard,
    )
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()

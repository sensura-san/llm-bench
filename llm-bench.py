#!/usr/bin/env python3
"""
LLM Benchmarking Script - Pythonic implementation with visualization
Runs llama-bench on multiple models and quantizations, then generates comparison graphs.
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Visualization imports
try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print(
        "Warning: matplotlib not installed. Graphs will not be generated.",
        file=sys.stderr,
    )
    print("Install with: pip install matplotlib", file=sys.stderr)


# Define once somewhere in __init__ or as a class-level constant
COMMON_QUANTS = [
    "F16",
    # --- Classic Q* baselines ---
    "Q4_0",  # classic 4-bit
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",  # near-lossless ref
    # --- K-block 3-bit family ---
    "Q3_K_S",  # smallest / most aggressive 3-bit K
    "Q3_K_M",  # middle 3-bit K
    "Q3_K_L",  # highest-quality 3-bit K
    # --- K-block 4/5/6-bit family ---
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    # --- IQ 2–4 bit family (selected) ---
    # "IQ2_XXS",  # very aggressive 2-bit
    # "IQ2_XS",
    # "IQ2_M",
    "IQ3_XS",
    "IQ3_M",
    "IQ4_NL",
]


@dataclass
class BenchConfig:
    """Benchmark configuration from environment variables"""

    out_dir: Path = field(default_factory=lambda: Path(os.getenv("OUT_DIR", "results")))
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", "models"))
    )
    llama_bench: Path = field(
        default_factory=lambda: Path(
            os.getenv("LLAMA", "~/.local/bin/llama-bench")
        ).expanduser()
    )
    threads: int = field(default_factory=lambda: int(os.getenv("THREADS", "4")))
    prompt_tokens: int = field(default_factory=lambda: int(os.getenv("PROMPT", "512")))
    gen_tokens: int = field(default_factory=lambda: int(os.getenv("GEN", "128")))
    repetitions: int = field(default_factory=lambda: int(os.getenv("REPS", "5")))
    cache_type_k: str = field(default_factory=lambda: os.getenv("CTYPE_K", ""))
    cache_type_v: str = field(default_factory=lambda: os.getenv("CTYPE_V", ""))
    models_filter: Optional[List[str]] = None
    quants_filter: Optional[List[str]] = None
    delete_after: bool = field(
        default_factory=lambda: os.getenv("DELETE_AFTER", "0") == "1"
    )
    cooldown_secs: int = field(
        default_factory=lambda: int(os.getenv("COOLDOWN_SECS", "120"))
    )
    generate_graphs: bool = field(
        default_factory=lambda: os.getenv("GENERATE_GRAPHS", "1") == "1"
    )

    def __post_init__(self):
        """Parse filter strings after initialization"""
        models_env = os.getenv("MODELS", "")
        self.models_filter = models_env.split() if models_env else None

        quants_env = os.getenv("QUANTS", "")
        self.quants_filter = quants_env.split() if quants_env else None

        # Model definitions: {path: [default_quantizations]}
        self.model_list = {
            self.model_dir / "qwen3-0.6b": COMMON_QUANTS,
            self.model_dir / "qwen2.5-0.5b-instruct": COMMON_QUANTS,
            self.model_dir / "gemma3-270m-instruct": COMMON_QUANTS,
            self.model_dir / "gemma3-1b-instruct": COMMON_QUANTS,
            self.model_dir / "llama3.2-1b": COMMON_QUANTS,
        }


@dataclass
class BenchResult:
    """Single benchmark result with all metrics"""

    test_time_utc: str
    model: str
    quant: str
    threads: int
    prompt_tps: Optional[float] = None
    gen_tps: Optional[float] = None
    pg_tps: Optional[float] = None
    cpu_info: str = "NA"
    gpu_info: str = "NA"
    cuda: str = "NA"
    vulkan: str = "NA"
    metal: str = "NA"
    sycl: str = "NA"
    gpu_blas: str = "NA"
    blas: str = "NA"
    flash_attn: str = "NA"
    n_gpu_layers: str = "NA"
    type_k: str = "NA"
    type_v: str = "NA"
    bench_opts: str = ""
    source: str = "file"

    def to_csv_row(self) -> List[str]:
        """Convert result to CSV row format"""

        def fmt(val):
            return str(val) if val is not None else "NA"

        return [
            self.test_time_utc,
            self.model,
            self.quant,
            str(self.threads),
            fmt(self.prompt_tps),
            fmt(self.gen_tps),
            fmt(self.pg_tps),
            self.cpu_info,
            self.gpu_info,
            self.cuda,
            self.vulkan,
            self.metal,
            self.sycl,
            self.gpu_blas,
            self.blas,
            self.flash_attn,
            self.n_gpu_layers,
            self.type_k,
            self.type_v,
            self.bench_opts,
            self.source,
        ]


class BenchmarkRunner:
    """Main benchmark orchestrator"""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.csv_path = config.out_dir / "results.csv"
        self.log_path = config.out_dir / "bench_errors.log"
        self.results: List[BenchResult] = []
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to file and console"""
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def validate_dependencies(self):
        """Verify llama-bench is available"""
        if not self.config.llama_bench.exists():
            self.logger.error(
                f"ERROR: llama-bench not found at {self.config.llama_bench}"
            )
            sys.exit(1)

        if not os.access(self.config.llama_bench, os.X_OK):
            self.logger.error(
                f"ERROR: llama-bench is not executable: {self.config.llama_bench}"
            )
            sys.exit(1)

    def init_csv(self):
        """Initialize CSV file with headers"""
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return  # File already exists with content

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "test_time_utc",
                    "model",
                    "quant",
                    "threads",
                    "prompt_tps",
                    "gen_tps",
                    "pg_tps",
                    "cpu_info",
                    "gpu_info",
                    "cuda",
                    "vulkan",
                    "metal",
                    "sycl",
                    "gpu_blas",
                    "blas",
                    "flash_attn",
                    "n_gpu_layers",
                    "type_k",
                    "type_v",
                    "bench_opts",
                    "source",
                ]
            )

    def append_result(self, result: BenchResult):
        """Append result to CSV and in-memory list"""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result.to_csv_row())
        self.results.append(result)

    def load_results_from_csv(self, csv_path: Optional[Path] = None):
        """Load results from a CSV file into self.results (for re-graphing only)"""
        if csv_path is None:
            csv_path = self.csv_path

        if not csv_path.exists():
            self.logger.error(f"No CSV file found at {csv_path}")
            return

        self.logger.info(f"Loading results from {csv_path}")
        self.results = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:

                def parse_float(val):
                    if val in (None, "", "NA"):
                        return None
                    try:
                        return float(val)
                    except ValueError:
                        return None

                result = BenchResult(
                    test_time_utc=row.get("test_time_utc", "NA"),
                    model=row.get("model", "NA"),
                    quant=row.get("quant", "NA"),
                    threads=int(row.get("threads", "0") or 0),
                    prompt_tps=parse_float(row.get("prompt_tps")),
                    gen_tps=parse_float(row.get("gen_tps")),
                    pg_tps=parse_float(row.get("pg_tps")),
                    cpu_info=row.get("cpu_info", "NA"),
                    gpu_info=row.get("gpu_info", "NA"),
                    cuda=row.get("cuda", "NA"),
                    vulkan=row.get("vulkan", "NA"),
                    metal=row.get("metal", "NA"),
                    sycl=row.get("sycl", "NA"),
                    gpu_blas=row.get("gpu_blas", "NA"),
                    blas=row.get("blas", "NA"),
                    flash_attn=row.get("flash_attn", "NA"),
                    n_gpu_layers=row.get("n_gpu_layers", "NA"),
                    type_k=row.get("type_k", "NA"),
                    type_v=row.get("type_v", "NA"),
                    bench_opts=row.get("bench_opts", ""),
                    source=row.get("source", "file"),
                )
                self.results.append(result)

    def find_model_file(self, model_dir: Path, quant: str) -> Optional[Tuple[str, str]]:
        """
        Find model file for quantization.
        Returns: (file_path, source_type) or None
        """
        # Case-insensitive search for GGUF files
        matching_files = [
            f for f in model_dir.glob("*.gguf") if quant.lower() in f.name.lower()
        ]

        if matching_files:
            # Select smallest file (lower memory variant)
            best = min(matching_files, key=lambda f: f.stat().st_size)
            return str(best), "file"

        # Check for HuggingFace stub
        hf_stub = model_dir / f"{quant}.hfpath"
        if hf_stub.exists():
            return hf_stub.read_text().strip(), "hf"

        return None

    def detect_quantizations(self, model_dir: Path) -> List[str]:
        """Auto-detect available quantizations from files"""
        quants = set()

        # Extract from GGUF filenames
        for gguf in model_dir.glob("*.gguf"):
            stem = gguf.stem
            # Assumes format: modelname-QUANT.gguf
            parts = stem.split("-")
            if len(parts) > 1:
                quants.add(parts[-1])

        # Add from HF stubs
        for stub in model_dir.glob("*.hfpath"):
            quants.add(stub.stem)

        return sorted(quants)

    def should_process_model(self, model_name: str) -> bool:
        """Check if model passes filter"""
        return not self.config.models_filter or model_name in self.config.models_filter

    def delete_hf_cache(self, file_path: str):
        """Remove HuggingFace cached file if DELETE_AFTER enabled"""
        if not self.config.delete_after or not file_path.startswith("hf://"):
            return

        cache_home = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
        cache_dir = cache_home / "llama.cpp"
        basename = file_path.split("/")[-1]

        for cached in cache_dir.rglob(basename):
            try:
                cached.unlink()
                self.logger.debug(f"Deleted cache: {cached}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {cached}: {e}")

    def run_llama_bench(self, file_path: str) -> Optional[List[dict]]:
        """Execute llama-bench and return parsed JSON results"""
        cmd = [
            str(self.config.llama_bench),
            "-m",
            file_path,
            "-p",
            str(self.config.prompt_tokens),
            "-n",
            str(self.config.gen_tokens),
            "-pg",
            f"{self.config.prompt_tokens},{self.config.gen_tokens}",
            "-r",
            str(self.config.repetitions),
            "-o",
            "json",
            "-t",
            str(self.config.threads),
        ]

        # Add KV cache quantization if specified
        if self.config.cache_type_k:
            cmd.extend(["--cache-type-k", self.config.cache_type_k])
        if self.config.cache_type_v:
            cmd.extend(["--cache-type-v", self.config.cache_type_v])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stderr:
                self.logger.debug(f"llama-bench stderr: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            self.logger.error("Benchmark timed out")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Benchmark failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from llama-bench: {e}")
            return None

    def parse_bench_results(
        self, bench_data: List[dict], model: str, quant: str, source: str
    ) -> BenchResult:
        """Extract metrics from benchmark JSON array"""

        def first_value(key: str, default="NA") -> str:
            """Get first non-None value for key"""
            for item in bench_data:
                if val := item.get(key):
                    return str(val)
            return default

        def extract_tps(
            prompt_check: bool, gen_check: bool, both: bool = False
        ) -> Optional[float]:
            """Extract tokens-per-second for specific test type"""
            for item in bench_data:
                n_prompt = item.get("n_prompt", 0) or 0
                n_gen = item.get("n_gen", 0) or 0

                if both and n_prompt > 0 and n_gen > 0:
                    return item.get("avg_ts")
                elif prompt_check and n_prompt > 0 and n_gen == 0:
                    return item.get("avg_ts")
                elif gen_check and n_gen > 0 and n_prompt == 0:
                    return item.get("avg_ts")
            return None

        # Extract test time (first available)
        test_time = first_value("test_time", datetime.now(timezone.utc).isoformat())

        # Build bench options string
        bench_opts = (
            f"bench:p={self.config.prompt_tokens};"
            f"n={self.config.gen_tokens};"
            f"r={self.config.repetitions};"
            f"ctk={self.config.cache_type_k or 'NA'};"
            f"ctv={self.config.cache_type_v or 'NA'}"
        )

        return BenchResult(
            test_time_utc=test_time,
            model=model,
            quant=quant,
            threads=self.config.threads,
            prompt_tps=extract_tps(True, False),
            gen_tps=extract_tps(False, True),
            pg_tps=extract_tps(False, False, both=True),
            cpu_info=first_value("cpu_info"),
            gpu_info=first_value("gpu_info"),
            cuda=first_value("cuda"),
            vulkan=first_value("vulkan"),
            metal=first_value("metal"),
            sycl=first_value("sycl"),
            gpu_blas=first_value("gpu_blas"),
            blas=first_value("blas"),
            flash_attn=first_value("flash_attn"),
            n_gpu_layers=first_value("n_gpu_layers"),
            type_k=first_value("type_k"),
            type_v=first_value("type_v"),
            bench_opts=bench_opts,
            source=source,
        )

    def save_json_result(self, bench_data: List[dict], model: str, quant: str):
        """Save raw benchmark JSON to file"""
        # Extract timestamp for filename
        timestamp = next(
            (item.get("test_time") for item in bench_data if "test_time" in item),
            "unknown",
        )

        # Sanitize timestamp for filename
        safe_time = timestamp.replace(":", "-").replace("+", "_")
        json_path = self.config.out_dir / f"{model}_{quant}_{safe_time}.json"

        json_path.write_text(json.dumps(bench_data, indent=2))

    def generate_visualizations(self):
        """Create comparison graphs from benchmark results"""
        if not HAS_MATPLOTLIB:
            self.logger.warning("Skipping graph generation (matplotlib not installed)")
            return

        if not self.results:
            self.logger.warning("No results to visualize")
            return

        self.logger.info("\nGenerating visualization graphs...")

        # Filter results with valid metrics
        valid_results = [
            r for r in self.results if r.prompt_tps or r.gen_tps or r.pg_tps
        ]

        if not valid_results:
            self.logger.warning("No valid results for graphing")
            return

        # Prepare data
        labels = [f"{r.model}\n{r.quant}" for r in valid_results]
        prompt_speeds = [r.prompt_tps if r.prompt_tps else 0 for r in valid_results]
        gen_speeds = [r.gen_tps if r.gen_tps else 0 for r in valid_results]
        combined_speeds = [r.pg_tps if r.pg_tps else 0 for r in valid_results]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("LLM Benchmark Results Comparison", fontsize=16, fontweight="bold")

        # Plot 1: Prompt Processing Speed
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(labels)), prompt_speeds, color="steelblue", alpha=0.7)
        ax1.set_title("Prompt Processing Speed", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Tokens/Second")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Plot 2: Generation Speed
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(labels)), gen_speeds, color="coral", alpha=0.7)
        ax2.set_title("Text Generation Speed", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Tokens/Second")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Plot 3: Combined (Prompt + Generation)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(
            range(len(labels)), combined_speeds, color="mediumseagreen", alpha=0.7
        )
        ax3.set_title(
            "Combined (Prompt + Generation) Speed", fontsize=12, fontweight="bold"
        )
        ax3.set_ylabel("Tokens/Second")
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha="right")
        ax3.grid(axis="y", alpha=0.3)
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Plot 4: All metrics comparison
        ax4 = axes[1, 1]
        x = range(len(labels))
        width = 0.25
        ax4.bar(
            [i - width for i in x],
            prompt_speeds,
            width,
            label="Prompt",
            color="steelblue",
            alpha=0.7,
        )
        ax4.bar(x, gen_speeds, width, label="Generation", color="coral", alpha=0.7)
        ax4.bar(
            [i + width for i in x],
            combined_speeds,
            width,
            label="Combined",
            color="mediumseagreen",
            alpha=0.7,
        )
        ax4.set_title("All Metrics Comparison", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Tokens/Second")
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save figure
        graph_path = self.config.out_dir / "benchmark_comparison.png"
        plt.savefig(graph_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Saved visualization: {graph_path}")
        plt.close()

    def run(self):
        """Main execution pipeline"""
        self.validate_dependencies()
        self.init_csv()

        self.logger.info("Starting LLM benchmark run")
        self.logger.info(f"Output directory: {self.config.out_dir}")
        self.logger.info(
            f"Configuration: {self.config.threads} threads, "
            f"{self.config.prompt_tokens}p + {self.config.gen_tokens}g tokens, "
            f"{self.config.repetitions} reps\n"
        )

        # Process each model
        for model_dir, default_quants in self.config.model_list.items():
            model_name = model_dir.name

            if not self.should_process_model(model_name):
                self.logger.info(f"Skipping {model_name} (filtered)")
                continue

            if not model_dir.is_dir():
                self.logger.warning(f"Model directory not found: {model_dir}")
                continue

            # Determine quantizations to test
            quants = (
                self.config.quants_filter
                or self.detect_quantizations(model_dir)
                or default_quants
            )

            if not quants:
                self.logger.warning(f"No quantizations found for {model_name}")
                continue

            # Benchmark each quantization
            for quant in quants:
                file_result = self.find_model_file(model_dir, quant)

                if not file_result:
                    self.logger.warning(f"File not found: {model_name} / {quant}")
                    continue

                file_path, source = file_result
                self.logger.info(f"==> {model_name} / {quant} [{source}]")

                # Run benchmark
                bench_data = self.run_llama_bench(file_path)

                if not bench_data:
                    self.logger.error(f"Benchmark failed for {model_name} / {quant}")
                    # Record failure
                    bench_opts = (
                        f"bench:p={self.config.prompt_tokens};"
                        f"n={self.config.gen_tokens};"
                        f"r={self.config.repetitions};"
                        f"ctk={self.config.cache_type_k or 'NA'};"
                        f"ctv={self.config.cache_type_v or 'NA'}"
                    )
                    failed = BenchResult(
                        test_time_utc="NA",
                        model=model_name,
                        quant=quant,
                        threads=self.config.threads,
                        bench_opts=bench_opts,
                        source=source,
                    )
                    self.append_result(failed)
                    self.delete_hf_cache(file_path)
                    continue

                # Parse and save
                result = self.parse_bench_results(bench_data, model_name, quant, source)
                self.append_result(result)
                self.save_json_result(bench_data, model_name, quant)

                # Log results
                self.logger.info(
                    f"  Prompt: {result.prompt_tps:.2f} t/s"
                    if result.prompt_tps
                    else "  Prompt: N/A"
                )
                self.logger.info(
                    f"  Generation: {result.gen_tps:.2f} t/s"
                    if result.gen_tps
                    else "  Generation: N/A"
                )
                self.logger.info(
                    f"  Combined: {result.pg_tps:.2f} t/s\n"
                    if result.pg_tps
                    else "  Combined: N/A\n"
                )

                self.delete_hf_cache(file_path)

                # Cooldown
                if self.config.cooldown_secs > 0:
                    self.logger.info(
                        f"Cooling down for {self.config.cooldown_secs}s..."
                    )
                    time.sleep(self.config.cooldown_secs)

        # Generate visualizations
        if self.config.generate_graphs:
            self.generate_visualizations()

        self.logger.info("\nBenchmark complete!")
        self.logger.info(f"Results CSV: {self.csv_path}")
        self.logger.info(f"Error log: {self.log_path}")
        self.logger.info(f"JSON files: {self.config.out_dir}/*.json")
        if self.config.generate_graphs and HAS_MATPLOTLIB:
            self.logger.info(f"Graphs: {self.config.out_dir}/benchmark_comparison.png")


def main():
    """Entry point"""
    config = BenchConfig()
    runner = BenchmarkRunner(config)

    # Usage:
    #   ./llm-bench.py                      -> full benchmark + graphs
    #   ./llm-bench.py --graphs-only        -> graphs from default results.csv
    #   ./llm-bench.py --graphs-only path/to/other_results.csv
    if len(sys.argv) > 1 and sys.argv[1] == "--graphs-only":
        # Optional CSV path argument
        if len(sys.argv) > 2:
            csv_path = Path(sys.argv[2])
        else:
            csv_path = runner.csv_path

        try:
            runner.load_results_from_csv(csv_path)
            if config.generate_graphs:
                runner.generate_visualizations()
            else:
                runner.logger.info(
                    "Graph generation disabled by config (GENERATE_GRAPHS=0)"
                )
        except Exception as e:
            print(f"\n\nFatal error during graph generation: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)
        return

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

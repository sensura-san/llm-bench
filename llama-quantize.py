#!/usr/bin/env python3
"""
LLAMA Quantization Wrapper Script

This script wraps llama-quantize to batch-process model quantizations.
It automatically generates all available quantization formats from a source F16 model.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# All available quantization formats (excluding aliases)
# Aliases excluded: Q3_K (→ Q3_K_M), Q4_K (→ Q4_K_M), Q5_K (→ Q5_K_M)
ALL_QUANTIZATIONS: List[str] = [
    "F32",
    "F16",
    "BF16",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q2_K_S",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ1_S",
    "IQ1_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4_MOE",
]

# Quantization precision hierarchy (higher number = higher precision/larger size)
# Used to prevent "quantizing up" which is not supported
QUANTIZATION_PRECISION: dict[str, int] = {
    "F32": 100,  # 32-bit float (highest precision)
    "F16": 90,  # 16-bit float
    "BF16": 90,  # 16-bit bfloat (same level as F16)
    "Q8_0": 80,  # 8-bit quantization
    "Q6_K": 60,  # 6-bit quantization
    "Q5_K_M": 52,  # 5-bit quantization (medium)
    "Q5_K_S": 51,  # 5-bit quantization (small)
    "Q5_0": 50,  # 5-bit quantization
    "Q5_1": 50,  # 5-bit quantization
    "Q4_K_M": 42,  # 4-bit quantization (medium)
    "Q4_K_S": 41,  # 4-bit quantization (small)
    "Q4_0": 40,  # 4-bit quantization
    "Q4_1": 40,  # 4-bit quantization
    "IQ4_NL": 40,  # 4-bit non-linear
    "IQ4_XS": 40,  # 4-bit extra small
    "MXFP4_MOE": 40,  # 4-bit MoE
    "Q3_K_L": 33,  # 3-bit quantization (large)
    "Q3_K_M": 32,  # 3-bit quantization (medium)
    "Q3_K_S": 31,  # 3-bit quantization (small)
    "IQ3_M": 30,  # 3-bit quantization mix
    "IQ3_S": 30,  # 3-bit quantization
    "IQ3_XS": 30,  # 3-bit quantization
    "IQ3_XXS": 30,  # 3-bit quantization
    "Q2_K": 20,  # 2-bit quantization
    "Q2_K_S": 20,  # 2-bit quantization (small)
    "IQ2_M": 20,  # 2-bit quantization
    "IQ2_S": 20,  # 2-bit quantization
    "IQ2_XS": 20,  # 2-bit quantization
    "IQ2_XXS": 20,  # 2-bit quantization
    "TQ2_0": 20,  # 2-bit ternarization
    "IQ1_M": 10,  # 1-bit quantization
    "IQ1_S": 10,  # 1-bit quantization
    "TQ1_0": 10,  # 1-bit ternarization
}

# Preset filter: Edit this list to only quantize specific formats
# Set to None or empty list to use all quantizations
PRESET_FILTER = [
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
# Example preset filters:
# PRESET_FILTER = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]  # Common formats
# PRESET_FILTER = ["Q4_K_M", "Q5_K_M"]  # Balanced quality/size


def print_help() -> None:
    """Print detailed help information."""
    help_text = """
LLAMA Quantization Wrapper
==========================

Usage:
    python llama_quantize_wrapper.py <model_name> [options]

Arguments:
    model_name          Base name of the model (without -F16 suffix)
                       Example: 'gemma3-270m-instruct'

Options:
    -i, --input-dir     Input directory containing the F16 model (default: current directory)
    -o, --output-dir    Output directory for quantized models (default: same as input)
    -s, --source-quant  Source quantization format (default: F16)
    -d, --dry-run       Show what would be done without executing
    -l, --list          List all available quantization formats
    -h, --help          Show this help message

Environment Variables:
    LLAMA_QUANT_FILTER  Comma-separated list of quantization formats to use
                       Example: export LLAMA_QUANT_FILTER="Q4_0,Q4_K_M,Q8_0"
                       Takes precedence over PRESET_FILTER in code

Examples:
    # Quantize all formats from current directory
    python llama_quantize_wrapper.py gemma3-270m-instruct

    # Quantize with specific input/output directories
    python llama_quantize_wrapper.py gemma3-270m-instruct -i models/7B -o models/7B/quantized

    # Use custom source quantization
    python llama_quantize_wrapper.py my-model -s BF16

    # Filter via environment variable
    export LLAMA_QUANT_FILTER="Q4_0,Q5_0,Q8_0"
    python llama_quantize_wrapper.py gemma3-270m-instruct

    # Dry run to see what would be executed
    python llama_quantize_wrapper.py gemma3-270m-instruct --dry-run

Notes:
    - The script looks for a model file with format: {model_name}-{source_quant}.gguf
    - Output files are named: {model_name}-{target_quant}.gguf
    - Existing files are automatically skipped
    - Quantization prefixes are case-sensitive (uppercase)
"""
    print(help_text)


def list_quantizations() -> None:
    """List all available quantization formats."""
    print("\nAvailable Quantization Formats:")
    print("=" * 50)
    for quant in ALL_QUANTIZATIONS:
        print(f"  - {quant}")
    print(f"\nTotal: {len(ALL_QUANTIZATIONS)} formats")
    print()


def get_filtered_quantizations() -> List[str]:
    """
    Get the list of quantizations to use, respecting filters.

    Priority:
    1. LLAMA_QUANT_FILTER environment variable
    2. PRESET_FILTER in code
    3. ALL_QUANTIZATIONS (no filter)

    Returns:
        List of quantization format strings
    """
    # Check environment variable first
    env_filter = os.environ.get("LLAMA_QUANT_FILTER")
    if env_filter:
        filtered = [q.strip().upper() for q in env_filter.split(",")]
        print(f"Using quantizations from LLAMA_QUANT_FILTER: {', '.join(filtered)}")
        return filtered

    # Check preset filter
    if PRESET_FILTER:
        print(f"Using preset filter: {', '.join(PRESET_FILTER)}")
        return PRESET_FILTER

    # No filter, use all
    print(f"Using all {len(ALL_QUANTIZATIONS)} quantization formats")
    return ALL_QUANTIZATIONS


def find_source_file(
    model_name: str, input_dir: Path, source_quant: str
) -> Optional[Path]:
    """
    Find the source model file.

    Args:
        model_name: Base model name
        input_dir: Directory to search
        source_quant: Source quantization format

    Returns:
        Path to source file, or None if not found
    """
    source_file = input_dir / f"{model_name}-{source_quant}.gguf"
    if source_file.exists():
        return source_file
    return None


def can_quantize_to(source_quant: str, target_quant: str) -> tuple[bool, str]:
    """
    Check if we can quantize from source to target format.

    Args:
        source_quant: Source quantization format
        target_quant: Target quantization format

    Returns:
        Tuple of (can_quantize, reason)
    """
    # Same format
    if source_quant == target_quant:
        return False, "same as source format"

    # Check if both formats are in our precision map
    source_precision = QUANTIZATION_PRECISION.get(source_quant)
    target_precision = QUANTIZATION_PRECISION.get(target_quant)

    # If we don't know the precision, allow it (unknown format)
    if source_precision is None or target_precision is None:
        return True, ""

    # Can't quantize to higher precision
    if target_precision > source_precision:
        return (
            False,
            f"cannot quantize up (source: {source_quant} @ {source_precision}, target: {target_quant} @ {target_precision})",
        )

    return True, ""


def quantize_model(
    model_name: str,
    input_dir: Path,
    output_dir: Path,
    source_quant: str,
    target_quant: str,
    dry_run: bool = False,
) -> bool:
    """
    Quantize a model to a specific format.

    Assumes:
      - source file exists
      - target quant is allowed (no "quantizing up")
      - output file does not already exist

    Returns:
        True if successful, False if failed
    """
    source_file = input_dir / f"{model_name}-{source_quant}.gguf"
    output_file = output_dir / f"{model_name}-{target_quant}.gguf"

    cmd = ["llama-quantize", str(source_file), str(output_file), target_quant]

    if dry_run:
        print(f"🔍 [DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    print(f"⚙️  Quantizing to {target_quant}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully created {output_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to quantize to {target_quant}")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Error: llama-quantize command not found. Is it in your PATH?")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch quantize GGUF models using llama-quantize", add_help=False
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Base name of the model (without quantization suffix)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Input directory containing the source model",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for quantized models (default: same as input)",
    )
    parser.add_argument(
        "-s",
        "--source-quant",
        type=str,
        default="F16",
        help="Source quantization format (default: F16)",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all available quantization formats",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show detailed help message"
    )

    args = parser.parse_args()

    # Handle help and list flags
    if args.help:
        print_help()
        return 0

    if args.list:
        list_quantizations()
        return 0

    # Validate model name
    if not args.model_name:
        print("❌ Error: model_name is required\n")
        print("Usage: python llama_quantize_wrapper.py <model_name> [options]")
        print("Use --help for detailed help")
        return 1

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find source file
    print(f"\n{'=' * 60}")
    print("LLAMA Quantization Batch Process")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Source format: {args.source_quant}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")

    source_file = find_source_file(args.model_name, args.input_dir, args.source_quant)
    if not source_file:
        print("\n❌ Error: Source file not found")
        print(
            f"   Expected: {args.input_dir / f'{args.model_name}-{args.source_quant}.gguf'}"
        )
        return 1

    print(f"Source file: {source_file.name}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be created")

    # Get filtered quantizations
    print()
    quantizations = get_filtered_quantizations()

    # Process quantizations
    print(f"\n{'=' * 60}")
    print(f"Processing {len(quantizations)} quantizations...")
    print(f"{'=' * 60}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, quant in enumerate(quantizations, 1):
        print(f"[{i}/{len(quantizations)}] {quant}")

        output_file = output_dir / f"{args.model_name}-{quant}.gguf"
        already_exists = output_file.exists()

        can_quantize, reason = can_quantize_to(args.source_quant, quant)

        if already_exists:
            print(f"⏭️  Skipping {quant}: Output file already exists")
            skip_count += 1
            print()
            continue

        if not can_quantize:
            print(f"⏭️  Skipping {quant}: {reason}")
            skip_count += 1
            print()
            continue

        if quantize_model(
            args.model_name,
            args.input_dir,
            output_dir,
            args.source_quant,
            quant,
            args.dry_run,
        ):
            success_count += 1
        else:
            fail_count += 1

        print()

    # Summary
    print(f"{'=' * 60}")
    print("Summary:")
    print(f"  ✅ Successfully quantized: {success_count}")
    print(f"  ⏭️  Skipped (existing/same): {skip_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"{'=' * 60}\n")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

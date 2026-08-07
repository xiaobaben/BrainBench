"""Prepare standardized BrainBench Foundational Analysis inputs.

The four EDF-based datasets are written as ``{dataset}_{id}.edf``. SEED-V
remains in CNT format and is written as ``SEED-V-{id}.cnt``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import edfio
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "core"

DATASET_ORDER = ("isruc", "bcic2020-3", "mental-arithmetic", "mumtaz2016", "seed-v")
BCIC_DATA_RECORD_DURATION = 1.0 / 64.0

ISRUC_SPECS: Tuple[Tuple[str, str], ...] = tuple(
    (f"{index}/{index}.rec", f"ISRUC_{index:02d}.edf") for index in range(1, 6)
)
BCIC_SPECS: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"Training set/Data_Sample{index:02d}.mat",
        f"BCIC2020-3_{index:02d}.edf",
    )
    for index in range(1, 6)
)
MENTAL_ARITHMETIC_SPECS: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"edf/Subject{index:02d}_1.edf",
        f"MentalArithmetic_{index + 1:02d}.edf",
    )
    for index in range(5)
)
MUMTAZ_SPECS: Tuple[Tuple[str, str], ...] = tuple(
    (f"files/H S{index} EC.edf", f"Mumtaz2016_{index:02d}.edf")
    for index in range(1, 6)
)
SEED_V_SPECS: Tuple[Tuple[str, str], ...] = (
    ("files/1_1_20180804.cnt", "SEED-V-01.cnt"),
    ("files/2_1_20180416.cnt", "SEED-V-02.cnt"),
    ("files/3_1_20180414.cnt", "SEED-V-03.cnt"),
    ("files/4_1_20180414.cnt", "SEED-V-04.cnt"),
    ("files/5_1_20180719.cnt", "SEED-V-05.cnt"),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of one file."""
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_writable(path: Path, overwrite: bool) -> None:
    """Reject an existing output unless overwrite was requested."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")


def copy_with_prefix(source: Path, output: Path, overwrite: bool) -> Dict[str, object]:
    """Copy an EDF/CNT file to its standardized benchmark name."""
    ensure_writable(output, overwrite)
    shutil.copy2(source, output)
    source_hash = sha256_file(source)
    output_hash = sha256_file(output)
    if source_hash != output_hash:
        raise RuntimeError(f"Copied file hash mismatch: {source} -> {output}")
    return {
        "source": str(source),
        "output": str(output),
        "mode": "copy",
        "size": output.stat().st_size,
        "sha256": output_hash,
    }


def load_bcic_mat(source: Path) -> Tuple[np.ndarray, List[str], float]:
    """Load BCIC2020-3 epochs as a continuous channel-by-time matrix."""
    mat = loadmat(str(source), squeeze_me=True, struct_as_record=False)
    if "epo_train" not in mat:
        raise ValueError(f"Unsupported BCIC MAT structure: {source}")

    epo = mat["epo_train"]
    data = np.asarray(epo.x, dtype=np.float64)
    channel_names = [str(name) for name in np.asarray(epo.clab).reshape(-1).tolist()]
    sampling_frequency = float(epo.fs)

    if data.ndim == 3:
        channel_axes = [axis for axis, size in enumerate(data.shape) if size == len(channel_names)]
        if not channel_axes:
            raise ValueError(
                f"Cannot align BCIC data shape {data.shape} with {len(channel_names)} channels."
            )
        data = np.moveaxis(data, channel_axes[0], 0)
        data = data.reshape(data.shape[0], -1)
    elif data.ndim == 2:
        if data.shape[0] == len(channel_names):
            pass
        elif data.shape[1] == len(channel_names):
            data = data.T
        else:
            raise ValueError(
                f"Cannot align BCIC data shape {data.shape} with {len(channel_names)} channels."
            )
    else:
        raise ValueError(f"Unsupported BCIC data shape: {data.shape}")

    return data, channel_names, sampling_frequency


def convert_bcic(source: Path, output: Path, overwrite: bool) -> Dict[str, object]:
    """Convert one BCIC2020-3 MAT file to the validated EDF representation."""
    ensure_writable(output, overwrite)
    data, channel_names, sampling_frequency = load_bcic_mat(source)
    signals = [
        edfio.EdfSignal(
            data[index],
            sampling_frequency=sampling_frequency,
            label=channel_name,
            physical_dimension="uV",
            physical_range=None,
        )
        for index, channel_name in enumerate(channel_names)
    ]
    edfio.Edf(signals, data_record_duration=BCIC_DATA_RECORD_DURATION).write(output)

    written = edfio.read_edf(output)
    if len(written.signals) != len(channel_names):
        raise RuntimeError(f"BCIC channel-count mismatch after conversion: {output}")
    for index, signal in enumerate(written.signals):
        if signal.label != channel_names[index]:
            raise RuntimeError(f"BCIC channel-order mismatch after conversion: {output}")
        if signal.sampling_frequency != sampling_frequency:
            raise RuntimeError(f"BCIC sampling-frequency mismatch after conversion: {output}")
        if signal.data.shape[0] != data.shape[1]:
            raise RuntimeError(f"BCIC sample-count mismatch after conversion: {output}")

    return {
        "source": str(source),
        "output": str(output),
        "mode": "mat_to_edf",
        "channels": len(channel_names),
        "samples": int(data.shape[1]),
        "sampling_frequency": sampling_frequency,
        "size": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def dataset_jobs(
    dataset_dirs: Mapping[str, Path],
) -> Dict[str, Tuple[Path, Tuple[Tuple[str, str], ...]]]:
    """Return source directories and rename specifications for all datasets."""
    return {
        "isruc": (dataset_dirs["isruc"], ISRUC_SPECS),
        "bcic2020-3": (dataset_dirs["bcic2020-3"], BCIC_SPECS),
        "mental-arithmetic": (dataset_dirs["mental-arithmetic"], MENTAL_ARITHMETIC_SPECS),
        "mumtaz2016": (dataset_dirs["mumtaz2016"], MUMTAZ_SPECS),
        "seed-v": (dataset_dirs["seed-v"], SEED_V_SPECS),
    }


def dataset_dirs_from_root(data_root: Path) -> Dict[str, Path]:
    """Resolve the public dataset layout from one user-provided root."""

    source_root = data_root.expanduser().resolve()
    return {
        "isruc": source_root / "isruc",
        "bcic2020-3": source_root / "bcic2020-3",
        "mental-arithmetic": source_root / "MentalArithmetic",
        "mumtaz2016": source_root / "mumtaz",
        "seed-v": source_root / "seedv",
    }


def prepare_inputs(
    dataset_dirs: Mapping[str, Path],
    output_dir: Path,
    datasets: Sequence[str],
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare all selected Core input files."""
    jobs = dataset_jobs(dataset_dirs)
    selected = [(name, jobs[name][0], jobs[name][1]) for name in datasets]
    missing = [
        source_dir / source_name
        for _name, source_dir, specs in selected
        for source_name, _output_name in specs
        if not (source_dir / source_name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(str(path) for path in missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for dataset_name, source_dir, specs in selected:
        for source_name, output_name in specs:
            source = source_dir / source_name
            output = output_dir / output_name
            if dataset_name == "bcic2020-3":
                result = convert_bcic(source, output, overwrite)
            else:
                result = copy_with_prefix(source, output, overwrite)
            result["dataset"] = dataset_name
            results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Prepare BrainBench Foundational Analysis inputs from downloaded datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Directory containing isruc, bcic2020-3, MentalArithmetic, mumtaz, "
            "and seedv as direct subdirectories."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run unified Core benchmark input preparation."""
    args = build_parser().parse_args(argv)
    datasets = list(DATASET_ORDER)
    dataset_dirs = dataset_dirs_from_root(args.data_root)
    results = prepare_inputs(dataset_dirs, DEFAULT_OUTPUT_DIR, datasets, overwrite=True)
    for result in results:
        print(
            f"{result['dataset']}\t{Path(result['source']).name}\t"
            f"{Path(result['output']).name}\t{result['mode']}"
        )
    print(f"overall=PASS\tfiles={len(results)}\toutput_dir={DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

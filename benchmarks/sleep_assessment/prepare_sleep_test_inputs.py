"""Prepare standardized BrainBench Sleep Assessment inputs.

This script standardizes the five public datasets used by NeuroBench-Sleep:

- ISRUC
- HMC
- SHHS1
- MASSSS3
- Physionet2018

Outputs are written as ``{dataset}_{id}.edf`` and ``{dataset}_{id}.npy``.
Users provide one downloaded-dataset root and the script writes the files used
by the public case JSONs under ``data/sleep`` by default.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import shutil
import warnings
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.etree import ElementTree as ET

import edfio
import mne
import numpy as np
from scipy.io import loadmat

try:
    import wfdb
except ImportError:  # Only required when PhysioNet2018 is selected.
    wfdb = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sleep"

EPOCH_SEC = 30.0
VALID_LABEL_IDS = {0, 1, 2, 3, 4}
DIGITAL_RANGE = (-32767, 32767)
FIXED_START_DATE = date(2000, 1, 1)
FIXED_START_TIME = time(0, 0, 0)

DATASET_ORDER = ("isruc", "hmc", "shhs", "mass", "physionet2018")

ISRUC_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("01", "1/1.rec", "1/1_1.txt"),
    ("02", "2/2.rec", "2/2_1.txt"),
    ("03", "3/3.rec", "3/3_1.txt"),
    ("04", "4/4.rec", "4/4_1.txt"),
    ("05", "5/5.rec", "5/5_1.txt"),
)

HMC_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("01", "SN001.edf", "SN001_sleepscoring.edf"),
    ("02", "SN002.edf", "SN002_sleepscoring.edf"),
    ("03", "SN003.edf", "SN003_sleepscoring.edf"),
    ("04", "SN004.edf", "SN004_sleepscoring.edf"),
    ("05", "SN005.edf", "SN005_sleepscoring.edf"),
)

SHHS_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("01", "shhs1-200001.edf", "shhs1-200001-profusion.xml"),
    ("02", "shhs1-200002.edf", "shhs1-200002-profusion.xml"),
    ("03", "shhs1-200003.edf", "shhs1-200003-profusion.xml"),
    ("04", "shhs1-200004.edf", "shhs1-200004-profusion.xml"),
    ("05", "shhs1-200005.edf", "shhs1-200005-profusion.xml"),
)

MASS_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("01", "01-03-0001 PSG.edf", "01-03-0001 Base.edf"),
    ("02", "01-03-0002 PSG.edf", "01-03-0002 Base.edf"),
    ("03", "01-03-0003 PSG.edf", "01-03-0003 Base.edf"),
    ("04", "01-03-0004 PSG.edf", "01-03-0004 Base.edf"),
    ("05", "01-03-0005 PSG.edf", "01-03-0005 Base.edf"),
)

PHYSIONET_SPECS: Tuple[Tuple[str, str], ...] = (
    ("01", "tr03-0005"),
    ("02", "tr03-0029"),
    ("03", "tr03-0052"),
    ("04", "tr03-0061"),
    ("05", "tr03-0078"),
)

HMC_STAGE_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

MASS_STAGE_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage R": 4,
}

SHHS_STAGE_MAP: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,
    "5": 4,
}

PHYSIONET_STAGE_MAP: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}


@dataclass(frozen=True)
class DatasetPaths:
    isruc: Path
    hmc: Path
    shhs: Path
    mass: Path
    physionet2018: Path


def dataset_paths_from_root(data_root: Path) -> DatasetPaths:
    """Resolve the five downloaded datasets from one public root."""

    source_root = data_root.expanduser().resolve()
    return DatasetPaths(
        isruc=source_root / "isruc",
        hmc=source_root / "hmc",
        shhs=source_root / "shhs",
        mass=source_root / "mass",
        physionet2018=source_root / "physionet2018",
    )


def required_source_files(
    dataset_paths: DatasetPaths,
    datasets: Sequence[str],
) -> List[Path]:
    """Return every raw file required before preparation starts."""

    required: Dict[str, List[Path]] = {
        "isruc": [
            dataset_paths.isruc / relative
            for _subject, edf_relative, label_relative in ISRUC_SPECS
            for relative in (edf_relative, label_relative)
        ],
        "hmc": [
            dataset_paths.hmc / relative
            for _subject, edf_name, scoring_name in HMC_SPECS
            for relative in (edf_name, scoring_name)
        ],
        "shhs": [
            dataset_paths.shhs / relative
            for _subject, edf_name, xml_name in SHHS_SPECS
            for relative in (edf_name, xml_name)
        ],
        "mass": [
            dataset_paths.mass / relative
            for _subject, psg_name, base_name in MASS_SPECS
            for relative in (psg_name, base_name)
        ],
        "physionet2018": [
            dataset_paths.physionet2018 / source_name / f"{source_name}.{extension}"
            for _subject, source_name in PHYSIONET_SPECS
            for extension in ("mat", "hea", "arousal")
        ],
    }
    return [path for dataset in datasets for path in required[dataset]]


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_writable(path: Path, overwrite: bool) -> None:
    """Guard overwrites for one output path."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash for one file."""
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_label_array(labels: np.ndarray, label_name: str) -> np.ndarray:
    """Validate one benchmark label array."""
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D label array for {label_name}, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError(f"Empty label array for {label_name}")
    invalid = sorted(set(int(x) for x in np.unique(labels)) - VALID_LABEL_IDS)
    if invalid:
        raise ValueError(f"Unexpected labels in {label_name}: {invalid}")
    return labels.astype(np.int64, copy=False)


def compare_npy_exact(output_path: Path, expected_path: Path) -> Dict[str, object]:
    """Compare two NPY files exactly."""
    actual = np.load(output_path, allow_pickle=False)
    expected = np.load(expected_path, allow_pickle=False)
    return {
        "mode": "exact_array",
        "shape_match": tuple(actual.shape) == tuple(expected.shape),
        "dtype_match": str(actual.dtype) == str(expected.dtype),
        "values_match": bool(np.array_equal(actual, expected)),
    }


def read_edf_info(edf_path: Path) -> Dict[str, object]:
    """Return a small EDF metadata summary."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different highpass filters.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different lowpass filters.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    return {
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(len(raw.ch_names)),
        "n_times": int(raw.n_times),
        "duration_min": float(raw.n_times / raw.info["sfreq"] / 60.0),
        "channel_names": list(raw.ch_names),
    }


def compare_edf_exact(output_path: Path, expected_path: Path) -> Dict[str, object]:
    """Compare two EDF-like binary files exactly."""
    return {
        "mode": "sha256",
        "hash_match": sha256_file(output_path) == sha256_file(expected_path),
        "size_match": output_path.stat().st_size == expected_path.stat().st_size,
    }


def compare_edf_structural(output_path: Path, expected_path: Path) -> Dict[str, object]:
    """Compare two EDF files structurally."""
    actual = read_edf_info(output_path)
    expected = read_edf_info(expected_path)
    return {
        "mode": "structural",
        "sfreq_match": actual["sfreq"] == expected["sfreq"],
        "n_channels_match": actual["n_channels"] == expected["n_channels"],
        "n_times_match": actual["n_times"] == expected["n_times"],
        "duration_match": abs(actual["duration_min"] - expected["duration_min"]) < 1e-9,
        "channel_names_match": actual["channel_names"] == expected["channel_names"],
    }


def write_labels(output_path: Path, labels: np.ndarray, overwrite: bool) -> None:
    """Write one NPY label file."""
    ensure_parent(output_path)
    ensure_writable(output_path, overwrite)
    np.save(output_path, validate_label_array(labels, output_path.name), allow_pickle=False)


def copy_binary_file(source_path: Path, output_path: Path, overwrite: bool) -> None:
    """Copy one binary source file to its benchmark target name."""
    ensure_parent(output_path)
    ensure_writable(output_path, overwrite)
    shutil.copy2(source_path, output_path)


def parse_isruc_labels(label_path: Path) -> np.ndarray:
    """Load one ISRUC label txt file and map 5 -> 4."""
    labels: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        value = int(float(text))
        labels.append(4 if value == 5 else value)
    return validate_label_array(np.asarray(labels, dtype=np.int64), label_path.name)


def parse_hmc_labels(scoring_path: Path) -> np.ndarray:
    """Extract HMC labels from the scoring EDF annotations."""
    ann = mne.read_annotations(str(scoring_path))
    labels = [HMC_STAGE_MAP[desc] for desc in ann.description if desc in HMC_STAGE_MAP]
    return validate_label_array(np.asarray(labels, dtype=np.int64), scoring_path.name)


def parse_shhs_labels(xml_path: Path) -> np.ndarray:
    """Extract SHHS labels from Profusion XML SleepStage nodes."""
    root = ET.parse(xml_path).getroot()
    labels: List[int] = []
    for node in root.iter():
        if not node.tag.endswith("SleepStage"):
            continue
        text = (node.text or "").strip()
        if not text:
            continue
        if text not in SHHS_STAGE_MAP:
            raise ValueError(f"Unexpected SHHS SleepStage value {text!r} in {xml_path}")
        labels.append(SHHS_STAGE_MAP[text])
    return validate_label_array(np.asarray(labels, dtype=np.int64), xml_path.name)


def extract_mass_stage_records(base_edf_path: Path) -> List[Tuple[float, int]]:
    """Return MASSSS3 stage records as ``(onset_sec, label_id)`` pairs."""
    ann = mne.read_annotations(str(base_edf_path))
    records: List[Tuple[float, int]] = []
    for onset, description in zip(ann.onset, ann.description):
        if description not in MASS_STAGE_MAP:
            continue
        records.append((float(onset), MASS_STAGE_MAP[description]))
    if not records:
        raise ValueError(f"No valid MASSSS3 stage annotations found in {base_edf_path}")
    return records


def parse_mass_labels(base_edf_path: Path) -> np.ndarray:
    """Extract MASSSS3 labels from Base EDF annotations, skipping unknown stages."""
    labels = [label_id for _onset_sec, label_id in extract_mass_stage_records(base_edf_path)]
    return validate_label_array(np.asarray(labels, dtype=np.int64), base_edf_path.name)


def crop_mass_psg(
    psg_path: Path,
    labels: np.ndarray,
    start_offset_sec: float,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Crop one MASSSS3 PSG file to the benchmark label duration and export EDF."""
    ensure_parent(output_path)
    ensure_writable(output_path, overwrite)
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    tmax = start_offset_sec + (labels.size * EPOCH_SEC) - (1.0 / float(raw.info["sfreq"]))
    cropped = raw.copy().crop(tmin=start_offset_sec, tmax=tmax)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mne.export.export_raw(str(output_path), cropped, fmt="edf", overwrite=True)


def parse_physionet_header(header_path: Path) -> Dict[str, object]:
    """Parse a PhysioNet2018 WFDB header."""
    lines = header_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty header file: {header_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Unexpected first header line in {header_path}: {lines[0]}")
    return {
        "record_name": first[0],
        "n_channels": int(first[1]),
        "sfreq": float(first[2]),
        "signal_len": int(first[3]),
        "signal_lines": lines[1:],
    }


def extract_physionet_stage_annotations(record_base: Path) -> List[Tuple[int, str]]:
    """Return ordered sleep-stage annotations from a PhysioNet2018 record."""
    if wfdb is None:
        raise ImportError("PhysioNet2018 preparation requires the 'wfdb' package")
    ann = wfdb.rdann(str(record_base), "arousal")
    records = [
        (int(sample), str(label))
        for sample, label in zip(ann.sample, ann.aux_note)
        if str(label) in PHYSIONET_STAGE_MAP
    ]
    if not records:
        raise ValueError(f"No PhysioNet2018 sleep-stage annotations found for {record_base}")
    return records


def build_physionet_labels(
    stage_annotations: Sequence[Tuple[int, str]],
    signal_len: int,
    sfreq: float,
) -> Tuple[np.ndarray, int, int]:
    """Build one PhysioNet2018 label array aligned to the first valid stage sample."""
    epoch_samples = int(round(EPOCH_SEC * sfreq))
    start_sample = int(stage_annotations[0][0])
    if start_sample < 0 or start_sample >= signal_len:
        raise ValueError(
            f"Invalid PhysioNet2018 stage start {start_sample} for signal_len={signal_len}"
        )

    n_epochs = (signal_len - start_sample) // epoch_samples
    if n_epochs <= 0:
        raise ValueError("No full PhysioNet2018 epochs remain after stage alignment.")

    labels = np.empty(n_epochs, dtype=np.int64)
    labels[:] = PHYSIONET_STAGE_MAP[stage_annotations[-1][1]]

    for index, (sample, stage_name) in enumerate(stage_annotations):
        begin = max(0, (int(sample) - start_sample) // epoch_samples)
        if index + 1 < len(stage_annotations):
            next_sample = int(stage_annotations[index + 1][0])
            end = max(0, (next_sample - start_sample) // epoch_samples)
        else:
            end = n_epochs
        begin = min(begin, n_epochs)
        end = min(max(end, begin), n_epochs)
        if begin < end:
            labels[begin:end] = PHYSIONET_STAGE_MAP[stage_name]

    end_sample = start_sample + n_epochs * epoch_samples
    return validate_label_array(labels, "Physionet2018"), start_sample, end_sample


def load_physionet_subject(subject_dir: Path) -> Dict[str, object]:
    """Load one PhysioNet2018 subject from the original directory."""
    if wfdb is None:
        raise ImportError("PhysioNet2018 preparation requires the 'wfdb' package")
    subject_id = subject_dir.name
    record_base = subject_dir / subject_id
    header = parse_physionet_header(subject_dir / f"{subject_id}.hea")
    signals = loadmat(str(subject_dir / f"{subject_id}.mat"), squeeze_me=True, struct_as_record=False)
    if "val" not in signals:
        raise ValueError(f"Unsupported PhysioNet2018 MAT structure: {subject_dir / (subject_id + '.mat')}")
    digital = np.asarray(signals["val"], dtype=np.int16)
    if digital.ndim != 2:
        raise ValueError(f"Expected 2D PhysioNet2018 matrix, got {digital.shape}")
    if digital.shape != (header["n_channels"], header["signal_len"]):
        raise ValueError(
            f"PhysioNet2018 MAT shape mismatch for {subject_id}: "
            f"{digital.shape} vs ({header['n_channels']}, {header['signal_len']})"
        )

    record_dig = wfdb.rdrecord(str(record_base), physical=False)
    if tuple(record_dig.d_signal.T.shape) != tuple(digital.shape):
        raise ValueError(f"WFDB digital shape mismatch for {subject_id}: {record_dig.d_signal.T.shape} vs {digital.shape}")
    if not np.array_equal(record_dig.d_signal.T.astype(np.int16, copy=False), digital):
        raise ValueError(f"WFDB digital values do not match MAT val for {subject_id}")

    channel_names = list(record_dig.sig_name)
    units = list(record_dig.units)
    gains = np.asarray(record_dig.adc_gain, dtype=np.float64)
    baselines = np.asarray(record_dig.baseline, dtype=np.float64)
    if len(channel_names) != digital.shape[0]:
        raise ValueError(f"PhysioNet2018 channel-name mismatch for {subject_id}")

    physical = (digital.astype(np.float64) - baselines[:, None]) / gains[:, None]
    return {
        "subject_id": subject_id,
        "sfreq": float(header["sfreq"]),
        "signal_len": int(header["signal_len"]),
        "digital": digital,
        "physical": physical,
        "channel_names": channel_names,
        "units": units,
    }


def ensure_physical_range(values: np.ndarray) -> Tuple[float, float]:
    """Return a valid EDF physical range for one signal."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        delta = max(abs(minimum) * 0.01, 1.0)
        minimum -= delta
        maximum += delta
    return minimum, maximum


def write_physionet_edf(
    output_path: Path,
    physical_data: np.ndarray,
    channel_names: Sequence[str],
    units: Sequence[str],
    sfreq: float,
    overwrite: bool,
) -> None:
    """Write one self-contained PhysioNet2018 EDF file from physical values."""
    ensure_parent(output_path)
    ensure_writable(output_path, overwrite)
    signals: List[edfio.EdfSignal] = []
    for index, channel_name in enumerate(channel_names):
        signals.append(
            edfio.EdfSignal(
                physical_data[index].astype(np.float64, copy=False),
                sampling_frequency=float(sfreq),
                label=channel_name,
                physical_dimension=units[index],
                physical_range=ensure_physical_range(physical_data[index]),
                digital_range=DIGITAL_RANGE,
            )
        )
    edf = edfio.Edf(
        signals,
        patient=edfio.Patient(),
        recording=edfio.Recording(startdate=FIXED_START_DATE),
        starttime=FIXED_START_TIME,
    )
    edf.write(output_path)


def validate_physionet_edf(
    output_path: Path,
    physical_data: np.ndarray,
    channel_names: Sequence[str],
    units: Sequence[str],
    sfreq: float,
) -> Dict[str, object]:
    """Validate one generated PhysioNet2018 EDF against its source metadata."""
    edf = edfio.read_edf(output_path)
    if len(edf.signals) != len(channel_names):
        return {"mode": "edf_readback", "signals_match": False}

    sample_positions_per_signal: List[List[int]] = []
    sample_match = True
    unit_match = True
    label_match = True
    sfreq_match = True
    length_match = True
    max_abs_error = 0.0

    for index, signal in enumerate(edf.signals):
        label_match = label_match and signal.label == channel_names[index]
        unit_match = unit_match and signal.physical_dimension == units[index]
        sfreq_match = sfreq_match and abs(signal.sampling_frequency - sfreq) < 1e-9
        length_match = length_match and signal.data.shape[0] == physical_data.shape[1]
        positions = sorted(
            {
                0,
                1,
                max(0, physical_data.shape[1] // 2),
                max(0, physical_data.shape[1] - 2),
                max(0, physical_data.shape[1] - 1),
            }
        )
        sample_positions_per_signal.append(positions)
        observed = np.asarray(signal.data[positions], dtype=np.float64)
        expected = np.asarray(physical_data[index, positions], dtype=np.float64)
        error = np.abs(observed - expected)
        max_abs_error = max(max_abs_error, float(np.max(error)))
        # Allow one quantization step plus a tiny float margin.
        quant_step = (signal.physical_range.max - signal.physical_range.min) / (
            signal.digital_range.max - signal.digital_range.min
        )
        sample_match = sample_match and bool(np.all(error <= (abs(quant_step) + 1e-9)))

    return {
        "mode": "edf_readback",
        "label_match": label_match,
        "unit_match": unit_match,
        "sfreq_match": sfreq_match,
        "length_match": length_match,
        "sample_match": sample_match,
        "max_abs_error": max_abs_error,
        "sample_positions": sample_positions_per_signal[:3],
    }


def collect_subject_summary(
    dataset_name: str,
    subject_id: str,
    output_edf_path: Path,
    output_npy_path: Path,
) -> Dict[str, object]:
    """Build one common summary block for a generated subject."""
    label_array = np.load(output_npy_path, allow_pickle=False)
    edf_info = read_edf_info(output_edf_path)
    return {
        "dataset": dataset_name,
        "subject_id": subject_id,
        "output_edf": str(output_edf_path),
        "output_npy": str(output_npy_path),
        "edf_info": edf_info,
        "label_info": {
            "shape": list(label_array.shape),
            "unique_labels": sorted(int(x) for x in np.unique(label_array)),
        },
    }


def summarize_status(checks: Mapping[str, object], warning_keys: Iterable[str] = ()) -> Tuple[str, List[str]]:
    """Convert raw check booleans into a status string and warning list."""
    warnings: List[str] = []
    failed: List[str] = []
    warning_set = set(warning_keys)
    for key, value in checks.items():
        if isinstance(value, bool) and not value:
            if key in warning_set:
                warnings.append(key)
            else:
                failed.append(key)
    if failed:
        return "fail", warnings + failed
    if warnings:
        return "warn", warnings
    return "pass", []


def validate_generated_pair(
    output_edf: Path,
    output_npy: Path,
    source_edf: Path | None = None,
) -> Dict[str, object]:
    """Validate public outputs without requiring private reference files."""

    labels = validate_label_array(
        np.load(output_npy, allow_pickle=False),
        output_npy.name,
    )
    edf_info = read_edf_info(output_edf)
    checks: Dict[str, object] = {
        "edf_has_channels": int(edf_info["n_channels"]) > 0,
        "edf_has_samples": int(edf_info["n_times"]) > 0,
        "labels_are_one_dimensional": labels.ndim == 1,
        "labels_are_non_empty": labels.size > 0,
    }
    copy_compare = None
    if source_edf is not None:
        copy_compare = compare_edf_exact(output_edf, source_edf)
        checks["copied_edf_hash_match"] = bool(copy_compare["hash_match"])
        checks["copied_edf_size_match"] = bool(copy_compare["size_match"])
    status, details = summarize_status(checks)
    return {
        "status": status,
        "details": details,
        "checks": checks,
        "copy_compare": copy_compare,
    }


def prepare_isruc(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare ISRUC EDF and labels."""
    results: List[Dict[str, object]] = []
    for subject_id, edf_rel, label_rel in ISRUC_SPECS:
        source_edf = source_dir / edf_rel
        source_label = source_dir / label_rel
        output_edf = output_dir / f"ISRUC_{subject_id}.edf"
        output_npy = output_dir / f"ISRUC_{subject_id}.npy"
        copy_binary_file(source_edf, output_edf, overwrite)
        write_labels(output_npy, parse_isruc_labels(source_label), overwrite)

        validation = validate_generated_pair(output_edf, output_npy, source_edf)
        summary = collect_subject_summary("ISRUC", subject_id, output_edf, output_npy)
        summary["source_files"] = {
            "edf": str(source_edf),
            "label": str(source_label),
        }
        summary["validation"] = validation
        results.append(summary)
    return results


def prepare_hmc(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare HMC EDF and labels."""
    results: List[Dict[str, object]] = []
    for subject_id, edf_name, scoring_name in HMC_SPECS:
        source_edf = source_dir / edf_name
        source_scoring = source_dir / scoring_name
        output_edf = output_dir / f"HMC_{subject_id}.edf"
        output_npy = output_dir / f"HMC_{subject_id}.npy"
        copy_binary_file(source_edf, output_edf, overwrite)
        write_labels(output_npy, parse_hmc_labels(source_scoring), overwrite)

        validation = validate_generated_pair(output_edf, output_npy, source_edf)
        summary = collect_subject_summary("HMC", subject_id, output_edf, output_npy)
        summary["source_files"] = {
            "edf": str(source_edf),
            "scoring": str(source_scoring),
        }
        summary["validation"] = validation
        results.append(summary)
    return results


def prepare_shhs(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare SHHS1 EDF and labels."""
    results: List[Dict[str, object]] = []
    for subject_id, edf_name, xml_name in SHHS_SPECS:
        source_edf = source_dir / edf_name
        source_xml = source_dir / xml_name
        output_edf = output_dir / f"SHHS1_{subject_id}.edf"
        output_npy = output_dir / f"SHHS1_{subject_id}.npy"
        copy_binary_file(source_edf, output_edf, overwrite)
        write_labels(output_npy, parse_shhs_labels(source_xml), overwrite)

        validation = validate_generated_pair(output_edf, output_npy, source_edf)
        summary = collect_subject_summary("SHHS1", subject_id, output_edf, output_npy)
        summary["source_files"] = {
            "edf": str(source_edf),
            "xml": str(source_xml),
        }
        summary["validation"] = validation
        results.append(summary)
    return results


def prepare_mass(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare MASSSS3 EDF and labels."""
    results: List[Dict[str, object]] = []
    for subject_id, psg_name, base_name in MASS_SPECS:
        source_psg = source_dir / psg_name
        source_base = source_dir / base_name
        output_edf = output_dir / f"MASSSS3_{subject_id}.edf"
        output_npy = output_dir / f"MASSSS3_{subject_id}.npy"
        stage_records = extract_mass_stage_records(source_base)
        start_offset_sec = float(stage_records[0][0])
        labels = validate_label_array(
            np.asarray([label_id for _onset_sec, label_id in stage_records], dtype=np.int64),
            source_base.name,
        )
        crop_mass_psg(source_psg, labels, start_offset_sec, output_edf, overwrite)
        write_labels(output_npy, labels, overwrite)

        validation = validate_generated_pair(output_edf, output_npy)
        summary = collect_subject_summary("MASSSS3", subject_id, output_edf, output_npy)
        summary["source_files"] = {
            "psg": str(source_psg),
            "base": str(source_base),
        }
        summary["alignment"] = {
            "crop_start_sec": start_offset_sec,
            "crop_start_min": start_offset_sec / 60.0,
        }
        summary["validation"] = validation
        results.append(summary)
    return results


def prepare_physionet(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool,
) -> List[Dict[str, object]]:
    """Prepare Physionet2018 EDF and labels."""
    results: List[Dict[str, object]] = []
    for subject_id, source_name in PHYSIONET_SPECS:
        subject_dir = source_dir / source_name
        output_edf = output_dir / f"Physionet2018_{subject_id}.edf"
        output_npy = output_dir / f"Physionet2018_{subject_id}.npy"
        payload = load_physionet_subject(subject_dir)
        stage_annotations = extract_physionet_stage_annotations(subject_dir / source_name)
        labels, start_sample, end_sample = build_physionet_labels(
            stage_annotations,
            signal_len=int(payload["signal_len"]),
            sfreq=float(payload["sfreq"]),
        )
        cropped_physical = np.asarray(payload["physical"][:, start_sample:end_sample], dtype=np.float64)
        write_physionet_edf(
            output_edf,
            physical_data=cropped_physical,
            channel_names=list(payload["channel_names"]),
            units=list(payload["units"]),
            sfreq=float(payload["sfreq"]),
            overwrite=overwrite,
        )
        write_labels(output_npy, labels, overwrite)

        edf_validation = validate_physionet_edf(
            output_edf,
            physical_data=cropped_physical,
            channel_names=list(payload["channel_names"]),
            units=list(payload["units"]),
            sfreq=float(payload["sfreq"]),
        )
        local_validation = validate_generated_pair(output_edf, output_npy)
        status, details = summarize_status(
            {
                **local_validation["checks"],
                "edf_label_match": bool(edf_validation["label_match"]),
                "edf_unit_match": bool(edf_validation["unit_match"]),
                "edf_sfreq_match": bool(edf_validation["sfreq_match"]),
                "edf_length_match": bool(edf_validation["length_match"]),
                "edf_sample_match": bool(edf_validation["sample_match"]),
            }
        )
        summary = collect_subject_summary("Physionet2018", subject_id, output_edf, output_npy)
        summary["source_files"] = {
            "subject_dir": str(subject_dir),
            "mat": str(subject_dir / f"{source_name}.mat"),
            "hea": str(subject_dir / f"{source_name}.hea"),
            "arousal": str(subject_dir / f"{source_name}.arousal"),
        }
        summary["alignment"] = {
            "label_start_sample_original": start_sample,
            "label_end_sample_original": end_sample,
        }
        summary["validation"] = {
            "status": status,
            "details": details,
            "edf_compare": edf_validation,
            "checks": local_validation["checks"],
        }
        results.append(summary)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Prepare BrainBench Sleep Assessment EDF/NPY inputs from downloaded datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Directory containing isruc, hmc, shhs, mass, and physionet2018 "
            "as direct subdirectories."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run unified Sleep benchmark input preparation."""
    args = build_parser().parse_args(argv)
    dataset_names = list(DATASET_ORDER)
    output_dir = DEFAULT_OUTPUT_DIR
    dataset_paths = dataset_paths_from_root(args.data_root)

    missing = [path for path in required_source_files(dataset_paths, dataset_names) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source files:\n" + "\n".join(str(path) for path in missing)
        )
    if "physionet2018" in dataset_names and wfdb is None:
        raise ImportError("PhysioNet2018 preparation requires the 'wfdb' package")

    all_results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []

    handlers = {
        "isruc": lambda: prepare_isruc(output_dir, dataset_paths.isruc, True),
        "hmc": lambda: prepare_hmc(output_dir, dataset_paths.hmc, True),
        "shhs": lambda: prepare_shhs(output_dir, dataset_paths.shhs, True),
        "mass": lambda: prepare_mass(output_dir, dataset_paths.mass, True),
        "physionet2018": lambda: prepare_physionet(
            output_dir, dataset_paths.physionet2018, True
        ),
    }

    for dataset_name in dataset_names:
        for summary in handlers[dataset_name]():
            all_results.append(summary)
            validation = summary["validation"]
            if validation["status"] == "fail":
                failures.append(
                    {
                        "dataset": summary["dataset"],
                        "subject_id": summary["subject_id"],
                        "details": validation["details"],
                    }
                )
            elif validation["status"] == "warn":
                warnings.append(
                    {
                        "dataset": summary["dataset"],
                        "subject_id": summary["subject_id"],
                        "details": validation["details"],
                    }
                )

    report = {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "output_dir": str(output_dir),
        "datasets": dataset_names,
        "subject_count": len(all_results),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "subjects": all_results,
    }

    for summary in all_results:
        print(
            f"{summary['dataset']}_{summary['subject_id']}\t"
            f"{summary['validation']['status']}\t"
            f"edf={Path(summary['output_edf']).name}\t"
            f"labels={summary['label_info']['shape']}"
        )
    print(
        f"overall={report['status']}\t"
        f"subjects={report['subject_count']}\t"
        f"failures={report['failure_count']}\t"
        f"warnings={report['warning_count']}"
    )

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

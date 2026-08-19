#!/usr/bin/env python3
"""Prepare standardized, MAT-free NeuroBench-Multi benchmark inputs.

Signal outputs are limited to EDF, CNT, and BDF. Labels and event sidecars are
plain numeric NPY arrays loadable with ``allow_pickle=False``. The generated
tree is flat and is committed only after every output has passed validation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from scipy.io import loadmat


SCRIPT_PATH = Path(__file__).resolve()
# This file is distributed under ``outputs/multi_agent/NeuroBench-Multi``.
# Generated data belong to the output bundle, while the Sleep source and
# standardized SHHS reference remain in the containing NeuroBench checkout.
OUTPUT_BUNDLE_ROOT = SCRIPT_PATH.parents[1]
SOURCE_PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SOURCE_ROOT = Path("/data/cyn/EEG_data")
# ``data/multi`` is the only persistent public-data root.  Per-case inputs
# belong in the evaluator runtime, never under this directory.
DEFAULT_OUTPUT_DIR = OUTPUT_BUNDLE_ROOT / "data" / "multi"
DEFAULT_SHHS_DIR = SOURCE_PROJECT_ROOT / "data" / "sleep" / "original" / "shhs"
DEFAULT_SLEEP_STANDARD_DIR = SOURCE_PROJECT_ROOT / "data" / "sleep"

DATASET_ORDER = ("DEAP", "SEED-VIG", "SEED-VII", "SHHS", "simultaneous")
DEFAULT_SUBJECTS = (1, 2, 3, 4, 5)
SIGNAL_SUFFIXES = {".edf", ".cnt", ".bdf"}
FORBIDDEN_OUTPUT_SUFFIXES = {".mat", ".xml", ".csv", ".tsv", ".vmrk", ".evt"}
# The five required benchmark ids stay fixed. SEED-VII sessions 2-4 also
# contain its two remaining native categories, appended without shifting them.
EMOTION_IDS = {
    "anger": 0,
    "disgust": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "fear": 5,
    "surprise": 6,
}
SHHS_STAGE_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3, "5": 4}
NPY_SCHEMAS: Dict[str, List[str]] = {
    "deap_labels": ["valence", "arousal", "dominance", "liking"],
    "seed_vig_perclos": ["perclos"],
    "seed_vii_trial_labels": [
        "trial_index",
        "global_trial_number",
        "trial_start_sec",
        "trial_end_sec",
        "emotion_id",
        "self_report_score",
    ],
    "seed_vii_continuous_labels": ["trial_index", "time_sec", "value"],
    "seed_vii_events": ["onset_sec", "duration_sec", "event_code", "channel"],
    "sleep_stage_labels": ["stage_id_per_30_sec_epoch"],
    "brainvision_events": ["onset_sec", "duration_sec", "event_code", "channel"],
    "fnirs_events": [
        "onset_sec",
        "source_sample_index",
        "source_field_1",
        "source_field_2",
        "source_field_3",
        "source_field_4",
        "source_field_5",
        "source_field_6",
        "source_field_7",
        "source_field_8",
    ],
}

def configure_runtime_cache() -> None:
    """Point MNE/Numba caches at a writable temporary directory."""
    uid = getattr(os, "getuid", lambda: 0)()
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / f"neurobench-numba-{uid}")
    )
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"neurobench-mpl-{uid}")
    )


configure_runtime_cache()


@dataclass
class Artifact:
    dataset: str
    role: str
    output_path: Path
    source_paths: List[str]
    status: str
    schema: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildContext:
    root: Path
    dry_run: bool
    artifacts: List[Artifact] = field(default_factory=list)
    output_names: set[str] = field(default_factory=set)

    def output(self, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError(f"Output must use the flat test_multi layout: {name}")
        return self.root / name

    def add(
        self,
        *,
        dataset: str,
        role: str,
        name: str,
        sources: Sequence[Path | str],
        status: str,
        schema: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        if name in self.output_names:
            raise RuntimeError(f"Duplicate planned output: {name}")
        self.output_names.add(name)
        path = self.output(name)
        self.artifacts.append(
            Artifact(
                dataset=dataset,
                role=role,
                output_path=path,
                source_paths=[str(item) for item in sources],
                status=status,
                schema=schema,
                metadata=dict(metadata or {}),
            )
        )
        return path


def resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_numeric_npy(path: Path, values: Any) -> None:
    array = np.asarray(values)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"NPY output must be a plain numeric ndarray: {path} ({array.dtype})")
    if array.size == 0:
        raise ValueError(f"Refusing to write an empty NPY array: {path}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"NPY output contains non-finite values: {path}")
    np.save(path, array, allow_pickle=False)


def copy_binary(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")


def matlab_strings(values: Any) -> List[str]:
    result: List[str] = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        item = value
        if isinstance(item, np.ndarray):
            if item.size == 1:
                item = item.item()
            elif item.dtype.kind in {"U", "S"}:
                item = "".join(str(part) for part in item.reshape(-1))
        result.append(str(item).strip())
    return result


def safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def select_sessions(paths: Sequence[Path], policy: str) -> List[Path]:
    ordered = sorted(paths, key=lambda path: path.name)
    return ordered if policy == "all" else ordered[:1]


def parse_datasets(raw: Sequence[str]) -> List[str]:
    aliases = {name.casefold(): name for name in DATASET_ORDER}
    selected: List[str] = []
    for value in raw:
        for token in value.split(","):
            key = token.strip().casefold()
            if not key:
                continue
            if key not in aliases:
                raise ValueError(f"Unsupported dataset: {token}")
            name = aliases[key]
            if name not in selected:
                selected.append(name)
    return selected or list(DATASET_ORDER)


def parse_subjects(raw: Sequence[str]) -> List[int]:
    subjects: List[int] = []
    for item in raw:
        value = int(item)
        if value <= 0:
            raise ValueError(f"Subject id must be positive: {item}")
        if value not in subjects:
            subjects.append(value)
    if not subjects:
        raise ValueError("At least one subject is required")
    return subjects


def write_edf_mixed(
    path: Path,
    channels: Sequence[Tuple[str, np.ndarray, float, str]],
    *,
    data_record_duration: float = 1.0,
) -> None:
    import edfio

    durations = [np.asarray(values).size / float(sampling_rate) for _, values, sampling_rate, _ in channels]
    target_records = int(math.ceil(max(durations) / data_record_duration - 1e-12))
    signals = []
    for name, values, sampling_rate, unit in channels:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite signal values for {name}: {path}")
        samples_per_record = float(sampling_rate) * data_record_duration
        rounded_samples_per_record = int(round(samples_per_record))
        if not math.isclose(samples_per_record, rounded_samples_per_record, abs_tol=1e-5):
            raise ValueError(
                f"Sampling rate {sampling_rate} is incompatible with EDF record duration "
                f"{data_record_duration}: {path}"
            )
        target_samples = target_records * rounded_samples_per_record
        if array.size > target_samples:
            raise ValueError(f"Signal duration exceeds computed EDF duration for {name}: {path}")
        if array.size < target_samples:
            array = np.pad(array, (0, target_samples - array.size), mode="edge")
        signals.append(
            edfio.EdfSignal(
                array,
                sampling_frequency=float(sampling_rate),
                label=name[:16],
                physical_dimension=unit,
                physical_range=None,
            )
        )
    edfio.Edf(signals, data_record_duration=data_record_duration).write(path)


def export_mne_raw_to_edf(raw: Any, destination: Path) -> None:
    import mne

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mne.export.export_raw(str(destination), raw, fmt="edf", overwrite=True)


def process_deap(source_dir: Path, ctx: BuildContext, subjects: Sequence[int]) -> None:
    raw_dir = source_dir / "data_original"
    label_dir = source_dir / "data_preprocessed_matlab"
    require_dir(raw_dir, "DEAP raw")
    require_dir(label_dir, "DEAP labels")
    for subject in subjects:
        sid = f"{subject:02d}"
        source_bdf = raw_dir / f"s{sid}.bdf"
        source_mat = label_dir / f"s{sid}.mat"
        require_file(source_bdf, f"DEAP subject {sid} BDF")
        require_file(source_mat, f"DEAP subject {sid} labels")
        out_bdf = ctx.add(
            dataset="DEAP",
            role="signal",
            name=f"DEAP_{sid}.bdf",
            sources=[source_bdf],
            status="copied",
        )
        out_npy = ctx.add(
            dataset="DEAP",
            role="labels",
            name=f"DEAP_{sid}.npy",
            sources=[source_mat],
            status="mat_to_npy",
            schema="deap_labels",
        )
        if ctx.dry_run:
            continue
        copy_binary(source_bdf, out_bdf)
        mat = loadmat(source_mat, variable_names=["labels"], squeeze_me=True)
        if "labels" not in mat:
            raise ValueError(f"DEAP labels key missing: {source_mat}")
        labels = np.asarray(mat["labels"], dtype=np.float64)
        if labels.shape != (40, 4):
            raise ValueError(f"DEAP labels must have shape (40, 4): {source_mat}")
        save_numeric_npy(out_npy, labels)


def process_seed_vig(
    source_dir: Path,
    ctx: BuildContext,
    subjects: Sequence[int],
    session_policy: str,
) -> None:
    raw_dir = source_dir / "Raw_Data"
    label_dir = source_dir / "perclos_labels"
    require_dir(raw_dir, "SEED-VIG raw")
    require_dir(label_dir, "SEED-VIG labels")
    for subject in subjects:
        sid = f"{subject:02d}"
        sessions = select_sessions(list(raw_dir.glob(f"{subject}_*.mat")), session_policy)
        if not sessions:
            raise FileNotFoundError(f"SEED-VIG subject {sid}: no raw MAT session")
        for source_mat in sessions:
            label_mat = label_dir / source_mat.name
            require_file(label_mat, f"SEED-VIG labels for {source_mat.name}")
            session = safe_token(source_mat.stem.split("_", 1)[1])
            base = f"SEEDVIG_{sid}_session-{session}"
            out_edf = ctx.add(
                dataset="SEED-VIG",
                role="signal",
                name=f"{base}.edf",
                sources=[source_mat],
                status="mat_to_edf",
                metadata={"native_sampling_rates_hz": {"EEG": 200.0, "EOG": 125.0}},
            )
            out_npy = ctx.add(
                dataset="SEED-VIG",
                role="labels",
                name=f"{base}.npy",
                sources=[label_mat],
                status="mat_to_npy",
                schema="seed_vig_perclos",
                metadata={"label_interval_sec": 8.0},
            )
            if ctx.dry_run:
                continue
            raw_mat = loadmat(source_mat, squeeze_me=True, struct_as_record=False)
            if "EEG" not in raw_mat or "EOG" not in raw_mat:
                raise ValueError(f"SEED-VIG MAT must contain EEG and EOG: {source_mat}")
            eeg_struct = raw_mat["EEG"]
            eog_struct = raw_mat["EOG"]
            eeg = np.asarray(eeg_struct.data, dtype=np.float64)
            eeg_names = matlab_strings(eeg_struct.chn)
            eeg_fs = float(np.asarray(eeg_struct.sample_rate).reshape(-1)[0])
            if eeg.shape[1] == len(eeg_names):
                eeg = eeg.T
            elif eeg.shape[0] != len(eeg_names):
                raise ValueError(f"SEED-VIG EEG channel mismatch: {source_mat} {eeg.shape}")
            eog_h = np.asarray(eog_struct.eog_h, dtype=np.float64).reshape(-1)
            eog_v = np.asarray(eog_struct.eog_v, dtype=np.float64).reshape(-1)
            eeg_duration = eeg.shape[1] / eeg_fs
            eog_fs = eog_h.size / eeg_duration
            if eog_h.size != eog_v.size or not math.isclose(eog_fs, 125.0, abs_tol=1e-6):
                raise ValueError(f"SEED-VIG EOG alignment mismatch: {source_mat}")
            channels = [
                (name, eeg[index], eeg_fs, "uV") for index, name in enumerate(eeg_names)
            ]
            channels.extend([("EOG-H", eog_h, eog_fs, "uV"), ("EOG-V", eog_v, eog_fs, "uV")])
            write_edf_mixed(out_edf, channels)
            labels_mat = loadmat(label_mat, variable_names=["perclos"], squeeze_me=True)
            perclos = np.asarray(labels_mat.get("perclos"), dtype=np.float64).reshape(-1)
            if perclos.size == 0:
                raise ValueError(f"SEED-VIG PERCLOS missing or empty: {label_mat}")
            save_numeric_npy(out_npy, perclos)


def parse_shhs_labels(xml_path: Path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    labels: List[int] = []
    for node in root.iter():
        if not node.tag.endswith("SleepStage"):
            continue
        value = (node.text or "").strip()
        if not value:
            continue
        if value not in SHHS_STAGE_MAP:
            raise ValueError(f"Unexpected SHHS stage {value!r}: {xml_path}")
        labels.append(SHHS_STAGE_MAP[value])
    if not labels:
        raise ValueError(f"No SHHS SleepStage values found: {xml_path}")
    return np.asarray(labels, dtype=np.int64)


def process_shhs(
    source_dir: Path,
    sleep_standard_dir: Path,
    ctx: BuildContext,
    subjects: Sequence[int],
) -> None:
    require_dir(source_dir, "SHHS raw")
    require_dir(sleep_standard_dir, "Sleep standard data")
    for subject in subjects:
        if subject > 5:
            raise ValueError("The shared Sleep SHHS profile contains subjects 1 through 5 only")
        sid = f"{subject:02d}"
        source_id = 200000 + subject
        source_edf = source_dir / f"shhs1-{source_id}.edf"
        source_xml = source_dir / f"shhs1-{source_id}-profusion.xml"
        reference_edf = sleep_standard_dir / f"SHHS1_{sid}.edf"
        reference_npy = sleep_standard_dir / f"SHHS1_{sid}.npy"
        for path, label in (
            (source_edf, "SHHS EDF"),
            (source_xml, "SHHS Profusion XML"),
            (reference_edf, "Sleep reference EDF"),
            (reference_npy, "Sleep reference NPY"),
        ):
            require_file(path, label)
        out_edf = ctx.add(
            dataset="SHHS",
            role="signal",
            name=f"SHHS1_{sid}.edf",
            sources=[source_edf, reference_edf],
            status="copied_sleep_standard",
        )
        out_npy = ctx.add(
            dataset="SHHS",
            role="labels",
            name=f"SHHS1_{sid}.npy",
            sources=[source_xml, reference_npy],
            status="xml_to_npy_sleep_standard",
            schema="sleep_stage_labels",
            metadata={"epoch_duration_sec": 30.0, "stage_ids": [0, 1, 2, 3, 4]},
        )
        if ctx.dry_run:
            continue
        copy_binary(source_edf, out_edf)
        labels = parse_shhs_labels(source_xml)
        save_numeric_npy(out_npy, labels)
        if sha256_file(out_edf) != sha256_file(reference_edf):
            raise RuntimeError(f"SHHS EDF differs from Sleep standard: {sid}")
        expected = np.load(reference_npy, allow_pickle=False)
        actual = np.load(out_npy, allow_pickle=False)
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise RuntimeError(f"SHHS labels differ from Sleep standard: {sid}")


def cnt_annotations(path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    import mne

    raw = mne.io.read_raw_cnt(str(path), preload=False, verbose="ERROR")
    onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
    descriptions = np.asarray([float(str(item).strip()) for item in raw.annotations.description])
    return onsets, descriptions, float(raw.info["sfreq"]), float(raw.n_times / raw.info["sfreq"])


def read_trigger_times(path: Path) -> Tuple[List[int], List[datetime]]:
    codes: List[int] = []
    times: List[datetime] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            codes.append(int(row[0]))
            times.append(datetime.fromisoformat(row[1].strip()))
    if not times:
        raise ValueError(f"No trigger timestamps found: {path}")
    return codes, times


def parse_save_info(path: Path) -> Tuple[List[str], np.ndarray]:
    emotions: List[str] = []
    scores: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            source_name = ",".join(row[:-1])
            normalized = source_name.replace("\\", "/").casefold()
            matches = [name for name in EMOTION_IDS if f"/{name}/" in normalized]
            if len(matches) != 1:
                raise ValueError(f"Cannot resolve SEED-VII emotion from {source_name!r}: {path}")
            emotions.append(matches[0])
            scores.append(float(row[-1]))
    if not emotions:
        raise ValueError(f"No SEED-VII trial metadata found: {path}")
    return emotions, np.asarray(scores, dtype=np.float64)


def parse_eye_start_datetime(date_text: str, time_text: str) -> datetime:
    date_value = date_text.strip().replace("/", "-")
    raw = f"{date_value} {time_text.strip()}"
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, pattern)
        except ValueError:
            pass
    raise ValueError(f"Unsupported eye-tracker start datetime: {raw!r}")


def numeric_text(value: str) -> float:
    text = value.strip().replace(",", ".")
    return float(text) if text else math.nan


def load_seed_vii_eye(
    tsv_path: Path,
    trigger_zero: datetime,
    sync_slope: float,
    sync_intercept: float,
    eeg_duration: float,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    selected_names = [
        "Gaze point X",
        "Gaze point Y",
        "Pupil diameter left",
        "Pupil diameter right",
    ]
    elapsed: List[float] = []
    columns: List[List[float]] = [[] for _ in selected_names]
    valid_left: List[float] = []
    valid_right: List[float] = []
    eye_start: Optional[datetime] = None
    with tsv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Recording timestamp", "Recording date", "Recording start time", *selected_names, "Validity left", "Validity right"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"SEED-VII eye TSV missing columns {missing}: {tsv_path}")
        for row in reader:
            timestamp = row.get("Recording timestamp", "").strip()
            if not timestamp:
                continue
            if eye_start is None:
                eye_start = parse_eye_start_datetime(row["Recording date"], row["Recording start time"])
            elapsed.append(float(timestamp) / 1_000_000.0)
            for target, name in zip(columns, selected_names):
                target.append(numeric_text(row.get(name, "")))
            valid_left.append(1.0 if row.get("Validity left", "").strip().casefold() == "valid" else 0.0)
            valid_right.append(1.0 if row.get("Validity right", "").strip().casefold() == "valid" else 0.0)
    if eye_start is None or not elapsed:
        raise ValueError(f"No eye samples found: {tsv_path}")
    eye_elapsed = np.asarray(elapsed, dtype=np.float64)
    trigger_offset = (trigger_zero - eye_start).total_seconds()
    eeg_times = sync_slope * (eye_elapsed - trigger_offset) + sync_intercept
    order = np.argsort(eeg_times, kind="stable")
    eeg_times = eeg_times[order]
    keep = np.concatenate(([True], np.diff(eeg_times) > 0))
    eeg_times = eeg_times[keep]
    raw_values = np.vstack(
        [np.asarray(values, dtype=np.float64)[order][keep] for values in columns]
        + [
            np.asarray(valid_left, dtype=np.float64)[order][keep],
            np.asarray(valid_right, dtype=np.float64)[order][keep],
        ]
    )
    in_range = (eeg_times >= 0.0) & (eeg_times <= eeg_duration)
    eeg_times = eeg_times[in_range]
    raw_values = raw_values[:, in_range]
    if eeg_times.size < 2:
        raise ValueError(f"Eye data do not overlap EEG: {tsv_path}")
    sampling_rate = 250.0
    # EDF has no per-signal fractional start offset. Anchor the synchronized
    # eye stream at EEG t=0 and use the first observed value for the sub-ms gap.
    grid_start = 0.0
    grid_end = min(float(eeg_times[-1]), eeg_duration)
    sample_count = int(math.floor((grid_end - grid_start) * sampling_rate)) + 1
    grid = grid_start + np.arange(sample_count, dtype=np.float64) / sampling_rate
    output = np.empty((6, sample_count), dtype=np.float64)
    for index in range(4):
        values = raw_values[index]
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            output[index] = 0.0
        else:
            output[index] = np.interp(grid, eeg_times[finite], values[finite])
    nearest = np.searchsorted(eeg_times, grid, side="left")
    nearest = np.clip(nearest, 0, eeg_times.size - 1)
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(grid - eeg_times[previous]) <= np.abs(eeg_times[nearest] - grid)
    nearest[choose_previous] = previous[choose_previous]
    output[4] = raw_values[4, nearest]
    output[5] = raw_values[5, nearest]
    return output, sampling_rate, {
        "first_sample_sec": float(grid[0]),
        "last_sample_sec": float(grid[-1]),
        "source_sample_count": int(raw_values.shape[1]),
        "output_sample_count": int(sample_count),
    }


def seed_vii_continuous_rows(
    continuous_mat: Path,
    session_number: int,
    trial_starts: np.ndarray,
    interval_sec: float = 4.0,
) -> np.ndarray:
    data = loadmat(continuous_mat, squeeze_me=True, struct_as_record=False)
    rows: List[List[float]] = []
    for trial_index, start in enumerate(trial_starts, start=1):
        global_trial = (session_number - 1) * 20 + trial_index
        key = str(global_trial)
        if key not in data:
            raise ValueError(f"Continuous label {key} missing: {continuous_mat}")
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)
        for label_index, value in enumerate(values):
            rows.append([float(trial_index), float(start + label_index * interval_sec), float(value)])
    return np.asarray(rows, dtype=np.float64)


def process_seed_vii(
    source_dir: Path,
    ctx: BuildContext,
    subjects: Sequence[int],
    session_policy: str,
) -> None:
    eeg_dir = source_dir / "EEG_raw"
    eye_dir = source_dir / "EYE_raw"
    info_dir = source_dir / "save_info"
    continuous_dir = source_dir / "continuous_labels"
    for directory, label in (
        (eeg_dir, "SEED-VII EEG"),
        (eye_dir, "SEED-VII eye tracking"),
        (info_dir, "SEED-VII trial info"),
        (continuous_dir, "SEED-VII continuous labels"),
    ):
        require_dir(directory, label)
    for subject in subjects:
        sid = f"{subject:02d}"
        sessions = select_sessions(list(eeg_dir.glob(f"{subject}_*.cnt")), session_policy)
        if not sessions:
            raise FileNotFoundError(f"SEED-VII subject {sid}: no CNT session")
        continuous_mat = continuous_dir / f"{subject}.mat"
        require_file(continuous_mat, f"SEED-VII subject {sid} continuous labels")
        for source_cnt in sessions:
            source_stem = source_cnt.stem
            parts = source_stem.split("_")
            if len(parts) < 3:
                raise ValueError(f"Unexpected SEED-VII session name: {source_cnt.name}")
            session_number = int(parts[-1])
            session = safe_token("_".join(parts[1:]))
            source_eye = eye_dir / f"{source_stem}.tsv"
            source_save = info_dir / f"{source_stem}_save_info.csv"
            source_triggers = info_dir / f"{source_stem}_trigger_info.csv"
            for path, label in (
                (source_eye, "eye TSV"),
                (source_save, "save info"),
                (source_triggers, "trigger info"),
            ):
                require_file(path, f"SEED-VII {label} for {source_stem}")
            base = f"SEEDVII_{sid}_session-{session}"
            out_cnt = ctx.add(
                dataset="SEED-VII",
                role="EEG signal",
                name=f"{base}_EEG.cnt",
                sources=[source_cnt],
                status="copied",
            )
            out_eye = ctx.add(
                dataset="SEED-VII",
                role="EyeTracking signal",
                name=f"{base}_EyeTracking.edf",
                sources=[source_eye, source_triggers, source_cnt],
                status="tsv_to_edf",
                metadata={
                    "channels": [
                        "GazePointX",
                        "GazePointY",
                        "PupilLeft",
                        "PupilRight",
                        "ValidLeft",
                        "ValidRight",
                    ]
                },
            )
            out_labels = ctx.add(
                dataset="SEED-VII",
                role="labels",
                name=f"{base}.npy",
                sources=[source_save, source_triggers, source_cnt],
                status="csv_cnt_to_npy",
                schema="seed_vii_trial_labels",
                metadata={"emotion_ids": EMOTION_IDS},
            )
            out_continuous = ctx.add(
                dataset="SEED-VII",
                role="continuous labels",
                name=f"{base}_continuous.npy",
                sources=[continuous_mat, source_triggers, source_cnt],
                status="mat_to_npy",
                schema="seed_vii_continuous_labels",
                metadata={"label_interval_sec": 4.0},
            )
            out_events = ctx.add(
                dataset="SEED-VII",
                role="events",
                name=f"{base}_events.npy",
                sources=[source_cnt],
                status="cnt_annotations_to_npy",
                schema="seed_vii_events",
                metadata={
                    "event_codes": {"trial_start": 1, "trial_end": 2},
                    "corresponding_signal": f"{base}_EEG.cnt",
                },
            )
            if ctx.dry_run:
                continue
            copy_binary(source_cnt, out_cnt)
            onsets, descriptions, eeg_fs, eeg_duration = cnt_annotations(source_cnt)
            trigger_codes, trigger_times = read_trigger_times(source_triggers)
            if len(onsets) != len(trigger_times) or not np.array_equal(
                descriptions.astype(np.int64), np.asarray(trigger_codes, dtype=np.int64)
            ):
                raise ValueError(f"SEED-VII CNT/CSV trigger mismatch: {source_stem}")
            relative_trigger = np.asarray(
                [(value - trigger_times[0]).total_seconds() for value in trigger_times],
                dtype=np.float64,
            )
            sync_slope, sync_intercept = np.polyfit(relative_trigger, onsets, 1)
            residual = onsets - (sync_slope * relative_trigger + sync_intercept)
            max_residual = float(np.max(np.abs(residual)))
            if max_residual > 0.005:
                raise ValueError(f"SEED-VII synchronization residual exceeds 5 ms: {max_residual}")
            eye_data, eye_fs, eye_meta = load_seed_vii_eye(
                source_eye,
                trigger_times[0],
                float(sync_slope),
                float(sync_intercept),
                eeg_duration,
            )
            eye_channels = [
                ("GazePointX", eye_data[0], eye_fs, "px"),
                ("GazePointY", eye_data[1], eye_fs, "px"),
                ("PupilLeft", eye_data[2], eye_fs, "mm"),
                ("PupilRight", eye_data[3], eye_fs, "mm"),
                ("ValidLeft", eye_data[4], eye_fs, "bool"),
                ("ValidRight", eye_data[5], eye_fs, "bool"),
            ]
            write_edf_mixed(out_eye, eye_channels)
            artifact = next(item for item in ctx.artifacts if item.output_path == out_eye)
            artifact.metadata.update(
                {
                    **eye_meta,
                    "sampling_rate_hz": eye_fs,
                    "sync_slope": float(sync_slope),
                    "sync_intercept_sec": float(sync_intercept),
                    "sync_max_residual_sec": max_residual,
                    "sync_rms_residual_sec": float(np.sqrt(np.mean(residual**2))),
                }
            )
            starts = onsets[descriptions.astype(np.int64) == 1]
            ends = onsets[descriptions.astype(np.int64) == 2]
            emotions, scores = parse_save_info(source_save)
            if not (starts.size == ends.size == len(emotions) == scores.size == 20):
                raise ValueError(f"SEED-VII expected 20 complete trials: {source_stem}")
            trial_index = np.arange(1, starts.size + 1, dtype=np.float64)
            global_trial = (session_number - 1) * 20 + trial_index
            emotion_ids = np.asarray([EMOTION_IDS[name] for name in emotions], dtype=np.float64)
            labels = np.column_stack((trial_index, global_trial, starts, ends, emotion_ids, scores))
            save_numeric_npy(out_labels, labels.astype(np.float64, copy=False))
            continuous = seed_vii_continuous_rows(continuous_mat, session_number, starts)
            save_numeric_npy(out_continuous, continuous)
            events = np.column_stack(
                (onsets, np.zeros_like(onsets), descriptions, np.zeros_like(onsets))
            ).astype(np.float64, copy=False)
            save_numeric_npy(out_events, events)


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Unsafe ZIP member {member.filename!r}: {zip_path}") from exc
        archive.extractall(destination)
    return destination


def zip_members(zip_path: Path, suffix: str) -> List[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(
            item.filename for item in archive.infolist() if not item.is_dir() and item.filename.casefold().endswith(suffix.casefold())
        )


def parse_brainvision_event_text(text: str, sampling_rate: float) -> Tuple[np.ndarray, Dict[str, int]]:
    records: List[Tuple[str, str, int, int, int]] = []
    for line in text.splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 5:
            continue
        records.append((fields[0].strip(), fields[1].strip(), int(fields[2]), int(fields[3]), int(fields[4] or 0)))
    if not records:
        raise ValueError("No BrainVision markers found")
    non_stimulus = sorted(
        {f"{kind}:{description}" for kind, description, *_ in records if re.fullmatch(r"S\s*\d+", description) is None}
    )
    fallback_map = {name: -(index + 1) for index, name in enumerate(non_stimulus)}
    rows: List[List[float]] = []
    for kind, description, position, size, channel in records:
        match = re.fullmatch(r"S\s*(\d+)", description)
        code = int(match.group(1)) if match else fallback_map[f"{kind}:{description}"]
        rows.append(
            [
                (position - 1) / sampling_rate,
                max(size, 1) / sampling_rate,
                float(code),
                float(channel),
            ]
        )
    return np.asarray(rows, dtype=np.float64), fallback_map


def parse_brainvision_events(vmrk_path: Path, sampling_rate: float) -> Tuple[np.ndarray, Dict[str, int]]:
    return parse_brainvision_event_text(
        vmrk_path.read_text(encoding="utf-8", errors="replace"), sampling_rate
    )


def brainvision_sampling_rate(text: str) -> float:
    match = re.search(r"^SamplingInterval=(.+)$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError("SamplingInterval missing from BrainVision header")
    return 1_000_000.0 / float(match.group(1).strip())


def canonical_fnirs_sampling_rate(value: float) -> float:
    return 125.0 / 12.0 if math.isclose(value, 125.0 / 12.0, abs_tol=1e-5) else value


def hdr_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"{key} not found in fNIRS header")
    return match.group(1).strip().strip('"')


def hdr_block(text: str, key: str) -> List[str]:
    lines = text.splitlines()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        if line.strip().startswith(f'{key}="#'):
            start = index + 1
            break
    if start is None:
        raise ValueError(f"{key} block not found in fNIRS header")
    result: List[str] = []
    for line in lines[start:]:
        if line.strip() == '#"':
            break
        if line.strip():
            result.append(line.strip())
    return result


def parse_fnirs_header(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sampling_rate = float(hdr_value(text, "SamplingRate"))
    wavelengths = [item for item in re.split(r"[\s,;]+", hdr_value(text, "Wavelengths")) if item][:2]
    sd_map = {
        (int(source), int(detector)): int(column) - 1
        for source, detector, column in re.findall(r"(\d+)-(\d+):(\d+)", hdr_value(text, "S-D-Key"))
    }
    mask: List[List[int]] = []
    for line in hdr_block(text, "S-D-Mask"):
        row = [int(token) for token in line.replace('"', "").split() if token in {"0", "1"}]
        if row:
            mask.append(row)
    active: List[Tuple[int, int, int]] = []
    for source, row in enumerate(mask, start=1):
        for detector, enabled in enumerate(row, start=1):
            if enabled and (source, detector) in sd_map:
                active.append((source, detector, sd_map[(source, detector)]))
    if len(wavelengths) != 2 or not active:
        raise ValueError(f"Incomplete fNIRS metadata: {path}")
    return {"sampling_rate": sampling_rate, "wavelengths": wavelengths, "active": active}


def export_fnirs(
    run_dir: Path,
    destination: Path,
    *,
    alignment_slope: float,
    alignment_intercept_sec: float,
    target_duration_sec: float,
) -> Tuple[float, int, int, Dict[str, float]]:
    hdr = next(iter(sorted(run_dir.glob("*.hdr"))), None)
    wl1_path = next(iter(sorted(run_dir.glob("*.wl1"))), None)
    wl2_path = next(iter(sorted(run_dir.glob("*.wl2"))), None)
    if hdr is None or wl1_path is None or wl2_path is None:
        raise FileNotFoundError(f"Missing fNIRS HDR/WL1/WL2 files: {run_dir}")
    meta = parse_fnirs_header(hdr)
    wl1 = np.atleast_2d(np.loadtxt(wl1_path, dtype=np.float64))
    wl2 = np.atleast_2d(np.loadtxt(wl2_path, dtype=np.float64))
    if wl1.shape != wl2.shape:
        raise ValueError(f"fNIRS wavelength shapes differ: {wl1.shape} vs {wl2.shape}")
    names: List[str] = []
    values: List[np.ndarray] = []
    for source, detector, column in meta["active"]:
        if column >= wl1.shape[1]:
            raise ValueError(f"fNIRS active column {column} exceeds {wl1.shape[1]}")
        names.extend(
            [
                f"S{source:02d}D{detector:02d}_{meta['wavelengths'][0]}",
                f"S{source:02d}D{detector:02d}_{meta['wavelengths'][1]}",
            ]
        )
        values.extend([wl1[:, column], wl2[:, column]])
    data = np.asarray(values, dtype=np.float64)
    sampling_rate = canonical_fnirs_sampling_rate(float(meta["sampling_rate"]))
    source_times = np.arange(data.shape[1], dtype=np.float64) / sampling_rate
    target_count = int(math.floor(target_duration_sec * sampling_rate)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / sampling_rate
    source_query = (target_times - alignment_intercept_sec) / alignment_slope
    aligned = np.vstack(
        [np.interp(source_query, source_times, channel) for channel in data]
    )
    # NIRScout WL1/WL2 files store raw light-intensity values, not electrical
    # voltage or derived HbO/HbR concentration. Preserve them as dimensionless
    # arbitrary units instead of letting MNE incorrectly label them as uV.
    channels = [
        (name, aligned[index], sampling_rate, "a.u.") for index, name in enumerate(names)
    ]
    write_edf_mixed(destination, channels, data_record_duration=0.96)
    return sampling_rate, len(names), aligned.shape[1], {
        "source_start_on_eeg_sec": float(alignment_intercept_sec),
        "source_end_on_eeg_sec": float(
            alignment_slope * source_times[-1] + alignment_intercept_sec
        ),
        "leading_edge_fill_sec": float(max(0.0, alignment_intercept_sec)),
        "trailing_edge_fill_sec": float(
            max(
                0.0,
                target_times[-1]
                - (alignment_slope * source_times[-1] + alignment_intercept_sec),
            )
        ),
    }


def parse_fnirs_event_text(text: str, sampling_rate: float) -> np.ndarray:
    rows: List[List[float]] = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        values = [float(token) for token in tokens]
        if len(values) > 9:
            raise ValueError(f"Unexpected fNIRS EVT width: {len(values)}")
        values.extend([0.0] * (9 - len(values)))
        rows.append([values[0] / sampling_rate, *values])
    if not rows:
        raise ValueError("No fNIRS events found")
    return np.asarray(rows, dtype=np.float64)


def parse_fnirs_events(path: Path, sampling_rate: float) -> np.ndarray:
    return parse_fnirs_event_text(
        path.read_text(encoding="utf-8", errors="replace"), sampling_rate
    )


def fit_clock_alignment(source_times: np.ndarray, target_times: np.ndarray) -> Dict[str, float]:
    if source_times.size != target_times.size or source_times.size < 2:
        raise ValueError("Clock alignment requires equal event counts and at least two events")
    slope, intercept = np.polyfit(source_times, target_times, 1)
    residual = target_times - (slope * source_times + intercept)
    return {
        "slope": float(slope),
        "intercept_sec": float(intercept),
        "max_residual_sec": float(np.max(np.abs(residual))),
        "rms_residual_sec": float(np.sqrt(np.mean(residual**2))),
        "event_count": int(source_times.size),
    }


def best_event_alignment(
    source_times: np.ndarray, eeg_events: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Find the EEG marker sequence corresponding to one fNIRS EVT sequence."""
    if source_times.size < 2:
        return None
    codes = sorted({int(value) for value in eeg_events[:, 1] if value > 0})
    candidates: List[Tuple[Tuple[int, ...], np.ndarray]] = []
    for width in (1, 2, 3):
        for selected_codes in itertools.combinations(codes, width):
            mask = np.isin(eeg_events[:, 1].astype(np.int64), selected_codes)
            sequence = np.sort(eeg_events[mask, 0])
            if source_times.size <= sequence.size <= source_times.size + 2:
                candidates.append((selected_codes, sequence))
    best: Optional[Dict[str, Any]] = None
    best_cost = math.inf
    for selected_codes, sequence in candidates:
        width = int(source_times.size)
        for start in range(sequence.size - width + 1):
            target = sequence[start : start + width]
            fit = fit_clock_alignment(source_times, target)
            cost = fit["max_residual_sec"] + 10.0 * abs(fit["slope"] - 1.0)
            if cost < best_cost:
                best_cost = cost
                best = {
                    **fit,
                    "eeg_event_codes": list(selected_codes),
                    "eeg_event_window_start": start,
                }
    return best


def match_simultaneous_runs(
    eeg_zip: Path,
    fnirs_zip: Path,
    eeg_members: Sequence[Path],
    fnirs_runs: Sequence[str],
) -> List[Tuple[Path, str, Dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    eeg_events: Dict[Path, np.ndarray] = {}
    fnirs_events: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(eeg_zip) as archive:
        for member in eeg_members:
            vhdr_text = archive.read(member.as_posix()).decode("utf-8", errors="replace")
            vmrk_text = archive.read(member.with_suffix(".vmrk").as_posix()).decode(
                "utf-8", errors="replace"
            )
            events, _ = parse_brainvision_event_text(
                vmrk_text, brainvision_sampling_rate(vhdr_text)
            )
            eeg_events[member] = events[events[:, 2] > 0][:, [0, 2]]
    with zipfile.ZipFile(fnirs_zip) as archive:
        for run in fnirs_runs:
            hdr_member = next(name for name in archive.namelist() if name.startswith(f"{run}/") and name.endswith(".hdr"))
            evt_member = next(name for name in archive.namelist() if name.startswith(f"{run}/") and name.endswith(".evt"))
            hdr_text = archive.read(hdr_member).decode("utf-8", errors="replace")
            sampling_rate = canonical_fnirs_sampling_rate(float(hdr_value(hdr_text, "SamplingRate")))
            events = parse_fnirs_event_text(
                archive.read(evt_member).decode("utf-8", errors="replace"), sampling_rate
            )
            fnirs_events[run] = events[:, 0]
    cost = np.full((len(eeg_members), len(fnirs_runs)), 1e9, dtype=np.float64)
    fits: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for eeg_index, eeg_member in enumerate(eeg_members):
        for fnirs_index, fnirs_run in enumerate(fnirs_runs):
            source = fnirs_events[fnirs_run]
            fit = best_event_alignment(source, eeg_events[eeg_member])
            if fit is None:
                continue
            fits[(eeg_index, fnirs_index)] = fit
            cost[eeg_index, fnirs_index] = fit["max_residual_sec"] + 10.0 * abs(
                fit["slope"] - 1.0
            )
    row_indices, column_indices = linear_sum_assignment(cost)
    if len(row_indices) != len(eeg_members):
        raise RuntimeError("Could not assign one fNIRS run to every selected EEG run")
    matched: List[Tuple[Path, str, Dict[str, Any]]] = []
    for row, column in sorted(zip(row_indices, column_indices)):
        if cost[row, column] >= 1e8:
            raise RuntimeError(
                f"No event-compatible fNIRS run for {eeg_members[row].as_posix()}"
            )
        fit = fits[(int(row), int(column))]
        if fit["max_residual_sec"] > 0.25:
            raise RuntimeError(
                f"Cross-modal alignment residual exceeds 250 ms for "
                f"{eeg_members[row].stem}/{fnirs_runs[column]}: {fit['max_residual_sec']}"
            )
        matched.append((eeg_members[row], fnirs_runs[column], fit))
    return matched


def process_simultaneous(
    source_dir: Path,
    ctx: BuildContext,
    subjects: Sequence[int],
    session_policy: str,
) -> None:
    raw_dir = source_dir / "raw"
    fnirs_root = raw_dir / "fNIRS"
    require_dir(raw_dir, "simultaneous EEG raw")
    require_dir(fnirs_root, "simultaneous fNIRS raw")
    with tempfile.TemporaryDirectory(prefix="neurobench-multi-sim-") as temp_name:
        temp_root = Path(temp_name)
        for subject in subjects:
            sid = f"{subject:02d}"
            vp = f"VP{subject:03d}"
            eeg_zip = raw_dir / f"{vp}.zip"
            fnirs_zip = fnirs_root / f"{vp}.zip"
            require_file(eeg_zip, f"simultaneous EEG ZIP for {vp}")
            require_file(fnirs_zip, f"simultaneous fNIRS ZIP for {vp}")
            eeg_members = select_sessions(
                [Path(name) for name in zip_members(eeg_zip, ".vhdr")], session_policy
            )
            eeg_zip_files = set(zip_members(eeg_zip, ""))
            for member in eeg_members:
                marker = member.with_suffix(".vmrk").as_posix()
                data_file = member.with_suffix(".eeg").as_posix()
                if marker not in eeg_zip_files or data_file not in eeg_zip_files:
                    raise FileNotFoundError(
                        f"Incomplete BrainVision run in {eeg_zip}: {member.as_posix()}"
                    )
            fnirs_zip_files = set(zip_members(fnirs_zip, ""))
            run_names = sorted(
                {Path(name).parts[0] for name in zip_members(fnirs_zip, ".hdr") if len(Path(name).parts) > 1}
            )
            if not eeg_members or not run_names:
                raise FileNotFoundError(f"No complete simultaneous runs for {vp}")
            for run in run_names:
                present_suffixes = {
                    Path(name).suffix.casefold()
                    for name in fnirs_zip_files
                    if name.startswith(f"{run}/")
                }
                missing_suffixes = sorted({".hdr", ".wl1", ".wl2", ".evt"} - present_suffixes)
                if missing_suffixes:
                    raise FileNotFoundError(
                        f"Incomplete fNIRS run {run} in {fnirs_zip}; missing {missing_suffixes}"
                    )
            matched_runs = match_simultaneous_runs(
                eeg_zip, fnirs_zip, eeg_members, run_names
            )
            planned: List[Tuple[Path, str, Dict[str, Any], Path, Path, Path, Path]] = []
            for member, fnirs_run, alignment in matched_runs:
                run = safe_token(member.stem)
                eeg_base = f"simultaneous_{sid}_run-{run}_EEG"
                fnirs_base = f"simultaneous_{sid}_run-{run}_fNIRS"
                eeg_edf = ctx.add(
                    dataset="simultaneous",
                    role="EEG signal",
                    name=f"{eeg_base}.edf",
                    sources=[f"{eeg_zip}::{member.as_posix()}"],
                    status="brainvision_to_edf",
                )
                eeg_npy = ctx.add(
                    dataset="simultaneous",
                    role="EEG events",
                    name=f"{eeg_base}.npy",
                    sources=[f"{eeg_zip}::{member.with_suffix('.vmrk').as_posix()}"],
                    status="vmrk_to_npy",
                    schema="brainvision_events",
                    metadata={"corresponding_signal": f"{eeg_base}.edf"},
                )
                fnirs_edf = ctx.add(
                    dataset="simultaneous",
                    role="fNIRS signal",
                    name=f"{fnirs_base}.edf",
                    sources=[f"{fnirs_zip}::{fnirs_run}/"],
                    status="nirs_to_edf_clock_aligned",
                    metadata={
                        "paired_eeg_run": member.stem,
                        "source_fnirs_run": fnirs_run,
                        "clock_alignment": alignment,
                    },
                )
                fnirs_npy = ctx.add(
                    dataset="simultaneous",
                    role="fNIRS events",
                    name=f"{fnirs_base}.npy",
                    sources=[f"{fnirs_zip}::{fnirs_run}/*.evt"],
                    status="evt_to_npy_clock_aligned",
                    schema="fnirs_events",
                    metadata={
                        "corresponding_signal": f"{fnirs_base}.edf",
                        "paired_eeg_run": member.stem,
                        "source_fnirs_run": fnirs_run,
                        "clock_alignment": alignment,
                    },
                )
                planned.append(
                    (
                        member,
                        fnirs_run,
                        alignment,
                        eeg_edf,
                        eeg_npy,
                        fnirs_edf,
                        fnirs_npy,
                    )
                )
            if ctx.dry_run:
                continue
            eeg_extract = safe_extract_zip(eeg_zip, temp_root / f"{vp}-eeg")
            fnirs_extract = safe_extract_zip(fnirs_zip, temp_root / f"{vp}-fnirs")
            import mne

            for member, fnirs_run, alignment, eeg_edf, eeg_npy, fnirs_edf, fnirs_npy in planned:
                vhdr = eeg_extract / member
                vmrk = vhdr.with_suffix(".vmrk")
                require_file(vhdr, "extracted BrainVision header")
                require_file(vmrk, "extracted BrainVision markers")
                raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose="ERROR")
                eeg_duration = float(raw.n_times / raw.info["sfreq"])
                export_mne_raw_to_edf(raw, eeg_edf)
                events, event_map = parse_brainvision_events(vmrk, float(raw.info["sfreq"]))
                save_numeric_npy(eeg_npy, events)
                artifact = next(item for item in ctx.artifacts if item.output_path == eeg_npy)
                artifact.metadata["negative_event_code_map"] = event_map
                run_dir = fnirs_extract / fnirs_run
                sampling_rate, channel_count, sample_count, edge_fill = export_fnirs(
                    run_dir,
                    fnirs_edf,
                    alignment_slope=alignment["slope"],
                    alignment_intercept_sec=alignment["intercept_sec"],
                    target_duration_sec=eeg_duration,
                )
                evt = next(iter(sorted(run_dir.glob("*.evt"))), None)
                if evt is None:
                    raise FileNotFoundError(f"fNIRS EVT missing: {run_dir}")
                fnirs_events = parse_fnirs_events(evt, sampling_rate)
                fnirs_events[:, 0] = (
                    alignment["slope"] * fnirs_events[:, 0]
                    + alignment["intercept_sec"]
                )
                save_numeric_npy(fnirs_npy, fnirs_events)
                artifact = next(item for item in ctx.artifacts if item.output_path == fnirs_edf)
                artifact.metadata.update(
                    {
                        "sampling_rate_hz": sampling_rate,
                        "channel_count": channel_count,
                        "sample_count": sample_count,
                        "target_eeg_duration_sec": eeg_duration,
                        **edge_fill,
                    }
                )


def signal_summary(path: Path) -> Dict[str, Any]:
    import mne

    suffix = path.suffix.casefold()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if suffix == ".edf":
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(str(path), preload=False, verbose="ERROR")
        elif suffix == ".cnt":
            raw = mne.io.read_raw_cnt(str(path), preload=False, verbose="ERROR")
        else:
            raise ValueError(f"Unsupported signal output: {path}")
    summary: Dict[str, Any] = {
        "channel_count": int(len(raw.ch_names)),
        "channel_names": list(raw.ch_names),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "sample_count": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "annotation_count": int(len(raw.annotations)),
    }
    if suffix in {".edf", ".bdf"}:
        import edfio

        container = edfio.read_edf(path) if suffix == ".edf" else edfio.read_bdf(path)
        units: Dict[str, List[str]] = {}
        for signal in container.signals:
            units.setdefault(str(signal.physical_dimension), []).append(str(signal.label))
        summary["native_sampling_rates_hz"] = sorted(
            {float(signal.sampling_frequency) for signal in container.signals}
        )
        summary["physical_units"] = units
    elif suffix == ".cnt":
        summary["physical_units"] = {
            "Neuroscan CNT amplitude (MNE converts calibrated values to V)": list(raw.ch_names)
        }
    return summary


def validate_outputs(ctx: BuildContext) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for artifact in ctx.artifacts:
        path = artifact.output_path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Generated output missing or empty: {path}")
        suffix = path.suffix.casefold()
        if suffix in FORBIDDEN_OUTPUT_SUFFIXES:
            raise ValueError(f"Forbidden output format generated: {path}")
        structure: Dict[str, Any]
        if suffix in SIGNAL_SUFFIXES:
            structure = signal_summary(path)
        elif suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.dtype.kind not in "biufc" or array.size == 0:
                raise ValueError(f"Invalid numeric NPY output: {path}")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"Non-finite NPY output: {path}")
            structure = {"shape": list(array.shape), "dtype": str(array.dtype)}
            if artifact.role.endswith("events") and array.ndim == 2:
                signal_name = ctx.root / artifact.metadata.get(
                    "corresponding_signal", path.with_suffix(".edf").name
                )
                if signal_name.is_file() and array.shape[1] > 0:
                    duration = signal_summary(signal_name)["duration_sec"]
                    if float(np.min(array[:, 0])) < 0 or float(np.max(array[:, 0])) > duration + 1.0:
                        raise ValueError(f"Event time lies outside signal duration: {path}")
        else:
            raise ValueError(f"Unexpected output suffix: {path}")
        records.append(
            {
                "dataset": artifact.dataset,
                "role": artifact.role,
                "output_path": path.name,
                "format": suffix.lstrip("."),
                "conversion_status": artifact.status,
                "source_paths": artifact.source_paths,
                "schema": NPY_SCHEMAS.get(artifact.schema or ""),
                "metadata": artifact.metadata,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "structure": structure,
            }
        )
    unexpected = [
        path for path in ctx.root.iterdir() if path.is_file() and path.suffix.casefold() in FORBIDDEN_OUTPUT_SUFFIXES
    ]
    if unexpected:
        raise ValueError("Forbidden files found: " + ", ".join(path.name for path in unexpected))
    return records


def write_manifest(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    subjects: Sequence[int],
    session_policy: str,
) -> Dict[str, Any]:
    # Build an explicit recording index so case builders can resolve public
    # sources without scanning filenames or relying on per-case sidecars.
    by_recording: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        dataset = str(record["dataset"])
        name = str(record["output_path"])
        if dataset == "DEAP":
            recording_id = Path(name).stem
        elif dataset == "SEED-VIG":
            recording_id = Path(name).stem
        elif dataset == "SEED-VII":
            recording_id = Path(name).stem
            for suffix in ("_EEG", "_EyeTracking", "_continuous", "_events"):
                if recording_id.endswith(suffix):
                    recording_id = recording_id[: -len(suffix)]
                    break
        elif dataset == "SHHS":
            recording_id = Path(name).stem
        elif dataset == "simultaneous":
            recording_id = re.sub(r"_(EEG|fNIRS)$", "", Path(name).stem)
        else:
            raise ValueError(f"Unexpected standardized dataset in manifest: {dataset}")
        item = by_recording.setdefault((dataset, recording_id), {"recording_id": recording_id, "dataset": dataset, "artifacts": []})
        item["artifacts"].append({
            "role": record["role"], "path": name, "format": record["format"],
            "schema": record.get("schema"), "metadata": record.get("metadata", {}),
            "structure": record.get("structure", {}), "sha256": record["sha256"],
        })
    payload = {
        "schema_version": 4,
        "generated_by": "NeuroBench-Multi/preprocess_multi_data.py",
        "layout": "flat",
        "signal_formats": ["edf", "cnt", "bdf"],
        "sidecar_format": "numeric_npy_allow_pickle_false",
        "datasets": list(datasets),
        "subjects": [f"{value:02d}" for value in subjects],
        "session_policy": session_policy,
        "emotion_ids": EMOTION_IDS,
        "npy_schemas": NPY_SCHEMAS,
        "entry_count": len(records),
        "entries": list(records),
        "recordings": [by_recording[key] for key in sorted(by_recording)],
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_processors(
    ctx: BuildContext,
    source_dirs: Mapping[str, Path],
    sleep_standard_dir: Path,
    datasets: Sequence[str],
    subjects: Sequence[int],
    session_policy: str,
) -> None:
    if "DEAP" in datasets:
        process_deap(source_dirs["DEAP"], ctx, subjects)
    if "SEED-VIG" in datasets:
        process_seed_vig(source_dirs["SEED-VIG"], ctx, subjects, session_policy)
    if "SEED-VII" in datasets:
        process_seed_vii(source_dirs["SEED-VII"], ctx, subjects, session_policy)
    if "SHHS" in datasets:
        process_shhs(source_dirs["SHHS"], sleep_standard_dir, ctx, subjects)
    if "simultaneous" in datasets:
        process_simultaneous(source_dirs["simultaneous"], ctx, subjects, session_policy)


def commit_staging(staging: Path, output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        staging.rename(output_dir)
        return
    if not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Rerun with --overwrite.")
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except Exception:
        if not output_dir.exists() and backup.exists():
            backup.rename(output_dir)
        raise
    else:
        shutil.rmtree(backup)


def verify_against(output_dir: Path, reference_dir: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reference_dir = resolved(reference_dir)
    require_dir(reference_dir, "verification reference")
    checked = 0
    for record in records:
        relative = str(record["output_path"])
        actual = output_dir / relative
        expected = reference_dir / relative
        require_file(expected, f"reference output {relative}")
        if actual.suffix.casefold() == ".npy":
            left = np.load(actual, allow_pickle=False)
            right = np.load(expected, allow_pickle=False)
            matched = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        elif record["conversion_status"] in {"copied", "copied_sleep_standard"}:
            matched = sha256_file(actual) == sha256_file(expected)
        else:
            matched = signal_summary(actual) == signal_summary(expected)
        if not matched:
            raise RuntimeError(f"Verification mismatch: {relative}")
        checked += 1
    return {"status": "PASS", "reference_dir": str(reference_dir), "checked_files": checked}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare flat, MAT-free NeuroBench-Multi benchmark inputs."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deap-dir", type=Path)
    parser.add_argument("--seed-vig-dir", type=Path)
    parser.add_argument("--seed-vii-dir", type=Path)
    parser.add_argument("--shhs-dir", type=Path, default=DEFAULT_SHHS_DIR)
    parser.add_argument("--simultaneous-dir", type=Path)
    parser.add_argument("--sleep-standard-dir", type=Path, default=DEFAULT_SLEEP_STANDARD_DIR)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_ORDER),
        help="Space- or comma-separated subset of DEAP, SEED-VIG, SEED-VII, SHHS, simultaneous.",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=[str(value) for value in DEFAULT_SUBJECTS]
    )
    parser.add_argument("--session-policy", choices=("first", "all"), default="first")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--verify-against", type=Path)
    return parser


def print_report(report: Mapping[str, Any], print_json: bool) -> None:
    if print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(
        f"overall={report['status']}\tentries={report['entry_count']}\t"
        f"output_dir={report['output_dir']}"
    )
    for dataset, count in report["dataset_entry_counts"].items():
        print(f"{dataset}\tentries={count}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    source_root = resolved(args.source_root)
    datasets = parse_datasets(args.datasets)
    subjects = parse_subjects(args.subjects)
    output_dir = resolved(args.output_dir)
    source_dirs = {
        "DEAP": resolved(args.deap_dir or source_root / "DEAP"),
        "SEED-VIG": resolved(args.seed_vig_dir or source_root / "SEED-VIG"),
        "SEED-VII": resolved(args.seed_vii_dir or source_root / "SEED-VII"),
        "SHHS": resolved(args.shhs_dir),
        "simultaneous": resolved(
            args.simultaneous_dir or source_root / "simultaneous_EEG-NIRS"
        ),
    }
    sleep_standard_dir = resolved(args.sleep_standard_dir)
    planned = BuildContext(root=output_dir, dry_run=True)
    run_processors(
        planned,
        source_dirs,
        sleep_standard_dir,
        datasets,
        subjects,
        args.session_policy,
    )
    counts = {
        dataset: sum(item.dataset == dataset for item in planned.artifacts) for dataset in datasets
    }
    if args.dry_run:
        report = {
            "status": "DRY_RUN",
            "output_dir": str(output_dir),
            "entry_count": len(planned.artifacts),
            "dataset_entry_counts": counts,
            "planned_outputs": [item.output_path.name for item in planned.artifacts],
        }
        print_report(report, args.print_json)
        return
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Rerun with --overwrite.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent))
    )
    committed = False
    try:
        ctx = BuildContext(root=staging, dry_run=False)
        run_processors(
            ctx,
            source_dirs,
            sleep_standard_dir,
            datasets,
            subjects,
            args.session_policy,
        )
        records = validate_outputs(ctx)
        manifest = write_manifest(
            staging, records, datasets, subjects, args.session_policy
        )
        if manifest["entry_count"] != len(ctx.artifacts):
            raise RuntimeError("Manifest entry count does not match generated artifacts")
        commit_staging(staging, output_dir, args.overwrite)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    verification = None
    if args.verify_against is not None:
        verification = verify_against(output_dir, args.verify_against, records)
    report = {
        "status": "PASS",
        "output_dir": str(output_dir),
        "entry_count": len(records),
        "dataset_entry_counts": counts,
        "manifest": str(output_dir / "manifest.json"),
        "verification": verification,
    }
    print_report(report, args.print_json)


if __name__ == "__main__":
    main()

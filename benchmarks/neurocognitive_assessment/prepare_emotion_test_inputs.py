#!/usr/bin/env python3
"""Prepare BrainBench Neurocognitive Assessment inputs.

This file contains the FACED, REFED, COG-BCI, and MPD-DF preprocessing
flows plus the FACED trial metadata needed to rebuild FACED from official pkl
files. It does not require the split preprocess_*.py files at runtime.

Example:
    python main.py prepare neurocognitive_assessment \
        --data-root /path/to/emotion-data-root
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "downloads" / "raw" / "neurocognitive_assessment"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "emotion"
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "dataset" / "validation"

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "brainbench-mpl-cache")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message="EDF format requires equal-length data blocks.*",
    category=RuntimeWarning,
)




def add_subject_args(command: list[str], option: str, subjects: Sequence[str] | None) -> None:
    if subjects:
        command.extend([option, *subjects])


def progress_iter(items: Sequence[tuple[str, type, list[str]]]) -> Iterable[tuple[str, type, list[str]]]:
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="Preprocessing datasets", unit="dataset", file=sys.stdout)


def subject_progress_iter(items: Sequence[Any], desc: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, unit="subject", file=sys.stdout)


def run_with_argv(program: str, argv: Sequence[str], main_func: Any) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [program, *argv]
        main_func()
    finally:
        sys.argv = old_argv
class FacedPreprocessor:
    @staticmethod
    def main(argv: Sequence[str] | None = None) -> None:
        run_with_argv("preprocess_faced", argv or [], _faced_main)


class RefedPreprocessor:
    @staticmethod
    def main(argv: Sequence[str] | None = None) -> None:
        run_with_argv("preprocess_refed", argv or [], _refed_main)


class CogBciPreprocessor:
    @staticmethod
    def main(argv: Sequence[str] | None = None) -> None:
        run_with_argv("preprocess_cog_bci", argv or [], _cog_bci_main)


class MpdDfPreprocessor:
    @staticmethod
    def main(argv: Sequence[str] | None = None) -> None:
        run_with_argv("preprocess_mpd_df", argv or [], _mpd_df_main)


# FacedPreprocessor implementation copied from data/preprocess_faced.py.
import argparse
import csv
import importlib
import json
import math
import pickle
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
_faced_DEFAULT_DATASET = 'FACED'
_faced_DEFAULT_PKL_ROOT = DEFAULT_DATASET_ROOT / 'FACED'
_faced_DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_faced_DEFAULT_VALIDATION_ROOT = DEFAULT_VALIDATION_ROOT
_faced_DEFAULT_METADATA_FILE = SCRIPT_DIR / 'faced_trial_metadata.json'
_faced_DEFAULT_TRIAL_COUNT = 28
_faced_DEFAULT_TRIAL_SECONDS = 30.0
_faced_DEFAULT_TARGET_SFREQ = 250.0
_faced_DEFAULT_SAMPLES_PER_TRIAL = int(_faced_DEFAULT_TRIAL_SECONDS * _faced_DEFAULT_TARGET_SFREQ)
_faced_MAX_SUBJECTS = 5
_faced_TRIAL_START_CODE = 101
_faced_TRIAL_END_CODE = 102
_faced_EXPECTED_CHANNEL_NAMES = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2']
_faced_COHORT1_CHANNEL_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 17, 16]
_faced_SCORE_ORDER = ['joy', 'tenderness', 'inspiration', 'amusement', 'anger', 'disgust', 'fear', 'sadness', 'arousal', 'valence', 'familiarity', 'liking']
_faced_LABEL_NAMES = ('valence', 'arousal', 'emotion')
_faced_EMOTION_RULE = 'vid 1..12 -> -1, vid 13..16 -> 0, vid 17..28 -> 1'
_faced_VIDEO_EMOTION_LABEL = {**{vid: -1 for vid in range(1, 13)}, **{vid: 0 for vid in range(13, 17)}, **{vid: 1 for vid in range(17, 29)}}

@dataclass(frozen=True)
class _faced_TrialWindow:
    video_id: int
    source_start_sample: int
    source_end_sample: int

@dataclass(frozen=True)
class _faced_SubjectOutputs:
    edf: Path
    labels: Mapping[str, Path]
    validation_json: Path

    def all_paths(self, include_validation: bool) -> List[Path]:
        paths = [self.edf, *self.labels.values()]
        if include_validation:
            paths.append(self.validation_json)
        return paths

def _faced_require_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {module_name}. Install it before running this script.') from exc

def _faced_tqdm_iter(items: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        tqdm = _faced_require_module('tqdm').tqdm
    except SystemExit:
        return items
    return tqdm(items, **kwargs)

def _faced_subject_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search('(\\d+)$', path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10 ** 9, path.name)

def _faced_subject_id_from_dir(subject_dir: Path) -> str:
    match = re.search('(\\d+)$', subject_dir.name)
    if not match:
        raise ValueError(f'Could not parse numeric subject id from {subject_dir.name}')
    return match.group(1)

def _faced_list_subject_dirs(source_root: Path) -> List[Path]:
    subjects = []
    for item in source_root.iterdir():
        if not item.is_dir():
            continue
        required = ['data.bdf', 'evt.bdf', 'After_remarks.mat', 'recordInformation.json']
        if all(((item / name).exists() for name in required)):
            subjects.append(item)
    return sorted(subjects, key=_faced_subject_sort_key)

def _faced_load_record_information(subject_dir: Path) -> Dict[str, Any]:
    info_path = subject_dir / 'recordInformation.json'
    with info_path.open('r', encoding='utf-8') as file_obj:
        return json.load(file_obj)

def _faced_load_recording_info(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'sub' not in (reader.fieldnames or []) or 'Cohort' not in (reader.fieldnames or []):
            return {}
        return {str(row['sub']): int(row['Cohort']) for row in reader}

def _faced_assert_edf_channel_name(name: str) -> None:
    try:
        name.encode('ascii')
    except UnicodeEncodeError as exc:
        raise AssertionError(f'EDF channel name is not ASCII: {name!r}') from exc
    if len(name) > 16:
        raise AssertionError(f'EDF channel name is longer than 16 chars: {name!r}')
    if any((ord(char) < 32 or ord(char) > 126 for char in name)):
        raise AssertionError(f'EDF channel name contains illegal characters: {name!r}')

def _faced_assert_all_edf_channel_names(ch_names: Sequence[str]) -> None:
    for name in ch_names:
        _faced_assert_edf_channel_name(name)

def _faced_assert_no_nonfinite_array(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        raise AssertionError(f'{context} contains NaN or Inf')

def _faced_assert_no_nonfinite_raw(raw: Any, context: str, block_seconds: float=60.0) -> None:
    sfreq = float(raw.info['sfreq'])
    block_samples = max(1, int(round(block_seconds * sfreq)))
    for start in range(0, int(raw.n_times), block_samples):
        stop = min(int(raw.n_times), start + block_samples)
        _faced_assert_no_nonfinite_array(raw.get_data(start=start, stop=stop), context)

def _faced_read_faced_raw(data_bdf: Path) -> Any:
    mne = _faced_require_module('mne')
    raw = mne.io.read_raw_bdf(str(data_bdf), preload=True, verbose='ERROR')
    return _faced_unit_check_like_faced_pkl(raw)

def _faced_unit_check_like_faced_pkl(raw: Any) -> Any:
    """Match the FACED pkl pipeline's unit_check: normalize raw data to uV."""
    checked = raw.copy()
    data_mean = float(np.mean(np.abs(checked._data)))
    if data_mean > 0 and math.log(data_mean) < 0:
        print('Unit change:', data_mean)
        checked._data = checked._data * 1000000.0
    return checked

def _faced_parse_numeric_trigger(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    match = re.search('-?\\d+', str(value).strip())
    if match is None:
        raise ValueError(f'Cannot parse numeric trigger from: {value}')
    return int(match.group())

def _faced_read_evt_arrays(evt_bdf: Path, sfreq: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _faced_read_evt_annotation_rows(evt_bdf)
    parsed_rows = []
    for description, onset_sec, duration_sec in rows:
        try:
            trigger = _faced_parse_numeric_trigger(description)
        except ValueError:
            continue
        parsed_rows.append((trigger, onset_sec, duration_sec))
    trigger = np.asarray([row[0] for row in parsed_rows], dtype=int)
    onset = np.asarray([int(round(row[1] * sfreq)) for row in parsed_rows], dtype=int)
    duration = np.asarray([int(round(row[2] * sfreq)) for row in parsed_rows], dtype=int)
    return (trigger, onset, duration)

def _faced_read_evt_annotation_rows(evt_bdf: Path) -> List[Tuple[str, float, float]]:
    mne = _faced_require_module('mne')
    annotations = mne.read_annotations(str(evt_bdf))
    return [(str(description).strip(), float(onset), float(duration)) for description, onset, duration in zip(annotations.description, annotations.onset, annotations.duration)]

def _faced_build_trial_windows_from_evt_annotations(evt_bdf: Path, sfreq: float, trial_count: int, trial_seconds: float) -> Dict[int, _faced_TrialWindow]:
    rows = _faced_read_evt_annotation_rows(evt_bdf)
    samples_per_trial = int(round(trial_seconds * sfreq))
    windows: Dict[int, _faced_TrialWindow] = {}
    for idx in range(len(rows) - 2):
        video_marker = rows[idx]
        start_marker = rows[idx + 1]
        end_marker = rows[idx + 2]
        trigger = video_marker[0]
        if not trigger.isdigit():
            continue
        video_id = int(trigger)
        if not 1 <= video_id <= trial_count:
            continue
        if start_marker[0] != str(_faced_TRIAL_START_CODE):
            continue
        if end_marker[0] != str(_faced_TRIAL_END_CODE):
            continue
        source_end = int(round(end_marker[1] * sfreq))
        source_start = source_end - samples_per_trial
        if source_start < 0:
            raise ValueError(f'Video {video_id} ends at sample {source_end}, less than {trial_seconds:g}s from file start.')
        windows[video_id] = _faced_TrialWindow(video_id=video_id, source_start_sample=source_start, source_end_sample=source_end)
    missing = [video_id for video_id in range(1, trial_count + 1) if video_id not in windows]
    if missing:
        first_events = [row[0] for row in rows[:40]]
        raise ValueError(f'Missing {_faced_TRIAL_END_CODE}-anchored FACED windows for videos: {missing}. First annotation descriptions: {first_events}')
    return windows

def _faced_load_mat_file(mat_path: Path) -> Dict[str, Any]:
    scipy_io = _faced_require_module('scipy.io')
    return scipy_io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

def _faced_usable_mat_items(mat: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in mat.items() if not key.startswith('__')}

def _faced_python_float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.item() if array.shape == () else value)

def _faced_python_int(value: Any) -> int:
    array = np.asarray(value)
    return int(array.item() if array.shape == () else value)

def _faced_load_trial_labels(mat_path: Path, trial_count: int) -> Tuple[Dict[str, np.ndarray], List[int]]:
    mat = _faced_load_mat_file(mat_path)
    if 'After_remark' not in mat:
        available = ', '.join(sorted(_faced_usable_mat_items(mat)))
        raise KeyError(f'After_remark not found in {mat_path}. Available keys: {available}')
    after_remarks = np.atleast_1d(mat['After_remark'])
    ratings_by_trial: Dict[int, Dict[str, float]] = {}
    seen_vids = set()
    for row_idx, remark in enumerate(after_remarks):
        trial_id = _faced_python_int(remark.trial)
        video_id = _faced_python_int(remark.vid)
        score = np.asarray(remark.score, dtype=np.float64).reshape(-1)
        if len(score) != len(_faced_SCORE_ORDER):
            raise ValueError(f'{mat_path} row {row_idx} has {len(score)} scores; expected {len(_faced_SCORE_ORDER)}')
        if trial_id in ratings_by_trial:
            raise ValueError(f'{mat_path} contains duplicate trial={trial_id}')
        if video_id in seen_vids:
            raise ValueError(f'{mat_path} contains duplicate vid={video_id}')
        seen_vids.add(video_id)
        ratings_by_trial[trial_id] = {'video_id': float(video_id), **{name: _faced_python_float(score[idx]) for idx, name in enumerate(_faced_SCORE_ORDER)}}
    expected_trials = list(range(1, trial_count + 1))
    missing_trials = [trial_id for trial_id in expected_trials if trial_id not in ratings_by_trial]
    if missing_trials:
        raise ValueError(f'{mat_path} is missing After_remark rows for trials: {missing_trials}')
    trial_video_ids = [int(ratings_by_trial[trial_id]['video_id']) for trial_id in expected_trials]
    missing_vids = [video_id for video_id in range(1, trial_count + 1) if video_id not in seen_vids]
    if missing_vids:
        raise ValueError(f'{mat_path} is missing After_remark rows for vids: {missing_vids}')
    valence = np.asarray([ratings_by_trial[trial_id]['valence'] for trial_id in expected_trials], dtype=np.float32)
    arousal = np.asarray([ratings_by_trial[trial_id]['arousal'] for trial_id in expected_trials], dtype=np.float32)
    emotion = np.asarray([_faced_VIDEO_EMOTION_LABEL[video_id] for video_id in trial_video_ids], dtype=np.float32)
    return ({'valence': valence, 'arousal': arousal, 'emotion': emotion}, trial_video_ids)

def _faced_load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def _faced_load_processed_pkl(path: Path, trial_count: int) -> np.ndarray:
    with path.open('rb') as handle:
        data = pickle.load(handle)
    array = np.asarray(data, dtype=np.float64)
    expected_shape = (trial_count, len(_faced_EXPECTED_CHANNEL_NAMES), _faced_DEFAULT_SAMPLES_PER_TRIAL)
    if array.shape != expected_shape:
        raise ValueError(f'{path}: expected shape {expected_shape}, got {array.shape}')
    _faced_assert_no_nonfinite_array(array, f'{path} pkl EEG')
    return array

def _faced_validation_trial_video_ids(validation: Mapping[str, Any], trial_count: int) -> List[int]:
    trials = list(validation['trials'])
    if len(trials) != trial_count:
        raise ValueError(f'expected {trial_count} validation trials, got {len(trials)}')
    trial_ids = [int(trial['trial_id']) for trial in trials]
    if trial_ids != list(range(1, trial_count + 1)):
        raise ValueError(f'trial_id order must be 1..{trial_count}, got {trial_ids}')
    video_ids = [int(trial['video_id']) for trial in trials]
    if sorted(video_ids) != list(range(1, trial_count + 1)):
        raise ValueError(f'video_id set must be 1..{trial_count}, got {video_ids}')
    return video_ids

def _faced_labels_from_validation(validation: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    trials = list(validation['trials'])
    return {'valence': np.asarray([trial['valence_mean'] for trial in trials], dtype=np.float32), 'arousal': np.asarray([trial['arousal_mean'] for trial in trials], dtype=np.float32), 'emotion': np.asarray([trial['emotion'] for trial in trials], dtype=np.float32)}

def _faced_rewrite_validation_from_pkl(validation: Mapping[str, Any], edf_name: str, trial_count: int) -> Dict[str, Any]:
    trials = []
    for index, trial in enumerate(validation['trials']):
        start_sample = index * _faced_DEFAULT_SAMPLES_PER_TRIAL
        end_sample = start_sample + _faced_DEFAULT_SAMPLES_PER_TRIAL
        rewritten = dict(trial)
        rewritten.update({'start_sample': int(start_sample), 'end_sample': int(end_sample), 'n_samples': int(_faced_DEFAULT_SAMPLES_PER_TRIAL), 'start_sec': float(start_sample / _faced_DEFAULT_TARGET_SFREQ), 'end_sec': float(end_sample / _faced_DEFAULT_TARGET_SFREQ), 'duration_sec': float(_faced_DEFAULT_TRIAL_SECONDS), 'label_points': 1})
        trials.append(rewritten)
    payload = dict(validation)
    payload.update({'edf': edf_name, 'sfreq': float(_faced_DEFAULT_TARGET_SFREQ), 'n_channels': len(_faced_EXPECTED_CHANNEL_NAMES), 'n_trials': int(trial_count), 'label_sfreq': float(1.0 / _faced_DEFAULT_TRIAL_SECONDS), 'trials': trials})
    _faced_assert_validation_json(payload, total_samples=trial_count * _faced_DEFAULT_SAMPLES_PER_TRIAL)
    return payload

def _faced_raw_from_processed_pkl(pkl_data_uv: np.ndarray, trial_video_ids: Sequence[int]) -> Any:
    mne = _faced_require_module('mne')
    ordered_uv = np.concatenate([pkl_data_uv[int(video_id) - 1] for video_id in trial_video_ids], axis=1)
    info = mne.create_info(_faced_EXPECTED_CHANNEL_NAMES, sfreq=_faced_DEFAULT_TARGET_SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(ordered_uv * 1e-06, info, first_samp=0, verbose='ERROR')
    raw.set_annotations(None)
    return raw

def _faced_parse_subject_index(value: str) -> int:
    match = re.search('\\d+', str(value))
    if match is None:
        raise ValueError(f'Cannot parse subject index from {value!r}')
    return int(match.group())

def _faced_list_pkl_subject_indices(pkl_root: Path, max_subjects: int) -> List[int]:
    indices = []
    for path in pkl_root.glob('sub*.pkl'):
        indices.append(_faced_parse_subject_index(path.stem))
    return sorted(indices)[:max_subjects]

def _faced_build_standardized_raw(source_raw: Any, evt_bdf: Path, windows: Mapping[int, _faced_TrialWindow], trial_video_ids: Sequence[int], trial_count: int, trial_seconds: float, target_sfreq: float, l_freq: float, h_freq: float, bad_thresh1: float, bad_proportion: float, cohort: int) -> Tuple[Any, int]:
    mne = _faced_require_module('mne')
    ICA = _faced_require_module('mne.preprocessing').ICA
    sfreq = float(source_raw.info['sfreq'])
    samples_per_trial = int(round(trial_seconds * sfreq))
    target_samples_per_trial = int(round(trial_seconds * target_sfreq))
    trigger, onset, duration = _faced_read_evt_arrays(evt_bdf, sfreq)
    events = np.transpose(np.vstack((np.vstack((onset, duration)), trigger)))
    epochs = mne.Epochs(source_raw, events, event_id=_faced_TRIAL_END_CODE, tmin=-trial_seconds, tmax=0, preload=True, verbose='ERROR')
    video_trigger_indices = np.where((trigger > 0) & (trigger < 29))[0]
    processed_by_video: Dict[int, np.ndarray] = {}
    for epoch_index, event_index in enumerate(video_trigger_indices):
        video_id = int(trigger[event_index])
        window = windows[video_id]
        expected_stop = window.source_start_sample + samples_per_trial
        if window.source_end_sample != expected_stop:
            raise AssertionError(f'Video {video_id} window length mismatch: {window.source_start_sample}:{window.source_end_sample}, expected {samples_per_trial} samples')
        if epoch_index >= len(epochs):
            raise AssertionError(f'Missing epoch for video {video_id}')
        processed_by_video[video_id] = _faced_preprocess_faced_epoch_like_pkl(epochs[epoch_index], target_sfreq=target_sfreq, l_freq=l_freq, h_freq=h_freq, bad_thresh1=bad_thresh1, bad_proportion=bad_proportion, target_samples_per_trial=target_samples_per_trial, ica_cls=ICA)
    missing = [video_id for video_id in range(1, trial_count + 1) if video_id not in processed_by_video]
    if missing:
        raise AssertionError(f'Missing processed videos: {missing}')
    pkl_data_uv = np.stack([processed_by_video[video_id] for video_id in range(1, trial_count + 1)], axis=0)
    pkl_data_uv = _faced_channel_modify_like_pkl(pkl_data_uv, cohort)
    standardized_data_uv = np.concatenate([pkl_data_uv[int(video_id) - 1] for video_id in trial_video_ids], axis=1)
    _faced_assert_no_nonfinite_array(standardized_data_uv, 'standardized EEG')
    standardized_data = standardized_data_uv * 1e-06
    _faced_assert_no_nonfinite_array(standardized_data, 'standardized EEG')
    info = mne.create_info(_faced_EXPECTED_CHANNEL_NAMES, sfreq=target_sfreq, ch_types='eeg')
    standardized_raw = mne.io.RawArray(standardized_data, info, first_samp=0, verbose='ERROR')
    standardized_raw.set_annotations(None)
    return (standardized_raw, target_samples_per_trial)

def _faced_channel_modify_like_pkl(pkl_data_uv: np.ndarray, cohort: int) -> np.ndarray:
    if cohort == 1:
        return pkl_data_uv[:, _faced_COHORT1_CHANNEL_ORDER, :]
    if cohort == 2:
        return pkl_data_uv
    print(f'Skip channel_modify for in-memory FACED pkl: cohort={cohort}')
    return pkl_data_uv

def _faced_preprocess_faced_epoch_like_pkl(epoch: Any, target_sfreq: float, l_freq: float, h_freq: float, bad_thresh1: float, bad_proportion: float, target_samples_per_trial: int, ica_cls: Any) -> np.ndarray:
    processed = epoch.copy()
    _faced_normalize_epoch_channel_metadata(processed)
    processed.resample(target_sfreq, verbose='ERROR')
    processed.filter(l_freq, h_freq, verbose='ERROR')
    _faced_interpolate_bad_channels_like_pkl(processed, bad_thresh1, bad_proportion)
    _faced_apply_ica_like_pkl(processed, ica_cls)
    processed.set_eeg_reference(ref_channels='average', verbose='ERROR')
    data_uv = np.squeeze(processed.get_data(copy=True))
    if data_uv.ndim != 2:
        raise AssertionError(f'Expected 2D epoch data, got shape {data_uv.shape}')
    if data_uv.shape[1] > target_samples_per_trial:
        data_uv = data_uv[:, -target_samples_per_trial:]
    elif data_uv.shape[1] < target_samples_per_trial:
        raise RuntimeError(f'The length of epoch is wrong: {data_uv.shape[1]} < {target_samples_per_trial}')
    return data_uv

def _faced_normalize_epoch_channel_metadata(epoch: Any) -> None:
    old_names = list(epoch.info['ch_names'])
    new_names = old_names.copy()
    if 'A1' not in new_names and len(new_names) >= 2:
        new_names[-2] = 'A2'
        new_names[-1] = 'A1'
    epoch.rename_channels({old: new for old, new in zip(old_names, new_names)})
    montage = _faced_require_module('mne').channels.make_standard_montage('standard_1020')
    epoch.set_montage(montage, verbose='ERROR')

def _faced_interpolate_bad_channels_like_pkl(epoch: Any, thresh1: float, proportion: float) -> None:
    data = np.squeeze(epoch.get_data(copy=True))
    if data.ndim != 2:
        raise AssertionError(f'Expected 2D epoch data for bad-channel check, got {data.shape}')
    median_abs = float(np.median(np.abs(data)))
    if median_abs <= 0:
        return
    bad_mask = np.abs(data) > thresh1 * median_abs
    bad_indices = np.flatnonzero(np.mean(bad_mask, axis=1) > proportion)
    if bad_indices.size == 0:
        print('No bad channel currently')
        return
    ch_names = list(epoch.info['ch_names'])
    bads = [ch_names[int(index)] for index in bad_indices]
    epoch.info['bads'].extend([name for name in bads if name not in epoch.info['bads']])
    print('Bad channels:', epoch.info['bads'])
    epoch.interpolate_bads(reset_bads=True, verbose='ERROR')

def _faced_apply_ica_like_pkl(epoch: Any, ica_cls: Any) -> None:
    ica = ica_cls(max_iter='auto', method='fastica')
    ica.fit(epoch, verbose='ERROR')
    eog_indices1, _ = ica.find_bads_eog(epoch, ch_name='Fp1', verbose='ERROR')
    eog_indices2, _ = ica.find_bads_eog(epoch, ch_name='Fp2', verbose='ERROR')
    ica.exclude = sorted(set(eog_indices1 + eog_indices2))
    ica.apply(epoch, verbose='ERROR')

def _faced_export_raw_to_edf(output_edf: Path, raw: Any, overwrite: bool) -> None:
    _faced_require_module('edfio')
    mne = _faced_require_module('mne')
    output_edf.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(output_edf), raw, fmt='edf', physical_range='channelwise', overwrite=overwrite)

def _faced_validate_exported_edf(output_edf: Path, expected_raw: Any, expected_n_times: int) -> None:
    mne = _faced_require_module('mne')
    reread = mne.io.read_raw_edf(str(output_edf), preload=False, verbose='ERROR')
    if len(reread.ch_names) != len(expected_raw.ch_names):
        raise AssertionError(f'{output_edf}: n_channels={len(reread.ch_names)}, expected {len(expected_raw.ch_names)}')
    if not math.isclose(float(reread.info['sfreq']), float(expected_raw.info['sfreq']), rel_tol=0.0, abs_tol=1e-09):
        raise AssertionError(f"{output_edf}: sfreq={reread.info['sfreq']}, expected {expected_raw.info['sfreq']}")
    if abs(int(reread.n_times) - int(expected_n_times)) > 1:
        raise AssertionError(f'{output_edf}: n_samples={reread.n_times}, expected {expected_n_times}')
    _faced_assert_all_edf_channel_names(reread.ch_names)
    reread.load_data(verbose='ERROR')
    _faced_assert_no_nonfinite_raw(reread, f'{output_edf} reread EEG')

def _faced_make_outputs(dataset: str, output_id: int, output_root: Path, validation_root: Path) -> _faced_SubjectOutputs:
    output_dir = output_root
    validation_dir = validation_root / dataset
    stem = f'{dataset}_{output_id:02d}'
    labels = {name: output_dir / f'{stem}_{name}.npy' for name in _faced_LABEL_NAMES}
    return _faced_SubjectOutputs(edf=output_dir / f'{stem}.edf', labels=labels, validation_json=validation_dir / f'{stem}_validation.json')

def _faced_save_labels(label_paths: Mapping[str, Path], labels: Mapping[str, np.ndarray], overwrite: bool) -> None:
    for name, path in label_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (not overwrite):
            print(f'SKIP existing label: {path}')
            continue
        array = np.asarray(labels[name], dtype=np.float32).reshape(-1)
        _faced_assert_no_nonfinite_array(array, f'{name} labels')
        np.save(path, array)
        print(f'WROTE label: {path}')

def _faced_build_validation_json(dataset: str, subject_id: str, edf_name: str, sfreq: float, n_channels: int, samples_per_trial: int, labels: Mapping[str, np.ndarray], trial_video_ids: Sequence[int], label_sfreq: float) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    for idx, video_id in enumerate(trial_video_ids):
        start_sample = idx * samples_per_trial
        end_sample = start_sample + samples_per_trial
        start_sec = float(start_sample / sfreq)
        end_sec = float(end_sample / sfreq)
        duration_sec = float(end_sec - start_sec)
        label_points = int(round(duration_sec * label_sfreq))
        trial = {'trial_id': int(idx + 1), 'original_trial': int(idx + 1), 'video_id': int(video_id), 'video_key': f'video_{video_id}', 'start_sample': int(start_sample), 'end_sample': int(end_sample), 'n_samples': int(end_sample - start_sample), 'start_sec': start_sec, 'end_sec': end_sec, 'duration_sec': duration_sec, 'label_points': label_points, 'valence_mean': float(labels['valence'][idx]), 'arousal_mean': float(labels['arousal'][idx]), 'emotion': int(labels['emotion'][idx])}
        trials.append(trial)
    validation = {'dataset': str(dataset), 'subject': str(subject_id), 'edf': str(edf_name), 'sfreq': float(sfreq), 'n_channels': int(n_channels), 'n_trials': int(len(trials)), 'label_type': 'static_valence_arousal_emotion', 'label_center': None, 'label_sfreq': float(label_sfreq), 'emotion_rule': _faced_EMOTION_RULE, 'trials': trials}
    _faced_assert_validation_json(validation, total_samples=len(trial_video_ids) * samples_per_trial)
    return validation

def _faced_assert_validation_json(validation: Mapping[str, Any], total_samples: int) -> None:
    trials = list(validation['trials'])
    if int(validation['n_trials']) != len(trials):
        raise AssertionError('n_trials must equal len(trials)')
    if not trials:
        raise AssertionError('validation JSON has no trials')
    if int(trials[0]['start_sample']) != 0:
        raise AssertionError('trials[0].start_sample must be 0')
    for prev, curr in zip(trials, trials[1:]):
        if int(prev['end_sample']) != int(curr['start_sample']):
            raise AssertionError('adjacent trials are not continuous')
    if int(trials[-1]['end_sample']) > int(total_samples):
        raise AssertionError('last trial exceeds EDF total samples')
    for trial in trials:
        if int(trial['n_samples']) != int(trial['end_sample']) - int(trial['start_sample']):
            raise AssertionError(f"trial {trial['trial_id']}: n_samples mismatch")
        duration = float(trial['end_sec']) - float(trial['start_sec'])
        if not math.isclose(float(trial['duration_sec']), duration, rel_tol=0.0, abs_tol=1e-09):
            raise AssertionError(f"trial {trial['trial_id']}: duration_sec mismatch")

def _faced_save_validation_json(path: Path, validation: Mapping[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not overwrite):
        print(f'SKIP existing validation JSON: {path}')
        return
    with path.open('w', encoding='utf-8') as file_obj:
        json.dump(validation, file_obj, indent=2, ensure_ascii=False)
        file_obj.write('\n')
    print(f'WROTE validation JSON: {path}')

def _faced_load_subject_metadata(metadata_file: Path, output_id: int, trial_count: int) -> Dict[str, Any]:
    metadata = _faced_load_json(metadata_file)
    subject_key = f'{output_id:02d}'
    try:
        subject_metadata = metadata['subjects'][subject_key]
    except KeyError as exc:
        raise FileNotFoundError(f'{metadata_file} does not contain FACED metadata for subject {subject_key}') from exc
    trials = []
    for trial in subject_metadata['trials']:
        trials.append({'trial_id': int(trial['trial_id']), 'video_id': int(trial['video_id']), 'valence_mean': float(trial['valence']), 'arousal_mean': float(trial['arousal']), 'emotion': int(trial['emotion'])})
    if len(trials) != trial_count:
        raise ValueError(f'{metadata_file} subject {subject_key} has {len(trials)} trials, expected {trial_count}')
    return {'trials': trials}

def _faced_assert_output_manifest(outputs: _faced_SubjectOutputs, include_validation: bool) -> None:
    missing = [str(path) for path in outputs.all_paths(include_validation) if not path.exists()]
    if missing:
        raise AssertionError(f'Missing expected output files: {missing}')

def _faced_process_subject(args: argparse.Namespace, subject_index: int) -> _faced_SubjectOutputs:
    output_id = subject_index + 1
    outputs = _faced_make_outputs(dataset=args.dataset, output_id=output_id, output_root=args.output_root, validation_root=args.validation_root)
    pkl_path = args.pkl_root / f'sub{subject_index:03d}.pkl'
    print(f'\nProcessing {pkl_path.name} -> {args.dataset}_{output_id:02d}')
    validation = _faced_load_subject_metadata(args.metadata_file, output_id, args.trial_count)
    trial_video_ids = _faced_validation_trial_video_ids(validation, trial_count=args.trial_count)
    labels = _faced_labels_from_validation(validation)
    pkl_data_uv = _faced_load_processed_pkl(pkl_path, trial_count=args.trial_count)
    standardized_raw = _faced_raw_from_processed_pkl(pkl_data_uv, trial_video_ids)
    samples_per_trial = _faced_DEFAULT_SAMPLES_PER_TRIAL
    if outputs.edf.exists() and (not args.overwrite):
        print(f'SKIP existing EDF: {outputs.edf}')
    else:
        _faced_export_raw_to_edf(outputs.edf, standardized_raw, overwrite=args.overwrite)
        print(f'WROTE EDF: {outputs.edf}')
    _faced_validate_exported_edf(output_edf=outputs.edf, expected_raw=standardized_raw, expected_n_times=standardized_raw.n_times)
    _faced_save_labels(outputs.labels, labels, overwrite=args.overwrite)
    if args.write_validation:
        label_sfreq = 1.0 / float(_faced_DEFAULT_TRIAL_SECONDS)
        validation = _faced_build_validation_json(dataset=args.dataset, subject_id=str(output_id), edf_name=outputs.edf.name, sfreq=float(standardized_raw.info['sfreq']), n_channels=len(standardized_raw.ch_names), samples_per_trial=samples_per_trial, labels=labels, trial_video_ids=trial_video_ids, label_sfreq=label_sfreq)
        validation = _faced_rewrite_validation_from_pkl(validation=validation, edf_name=outputs.edf.name, trial_count=args.trial_count)
        _faced_save_validation_json(outputs.validation_json, validation, overwrite=args.overwrite)
    _faced_assert_output_manifest(outputs, include_validation=args.write_validation)
    return outputs

def _faced_write_failed_log(path: Path, failures: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_obj:
        for subject, error in failures:
            file_obj.write(f'[{subject}]\n{error}\n\n')

def _faced_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild standardized NeuroBench FACED EDF/NPY files from official FACED pkl files.')
    parser.add_argument('--pkl-root', type=Path, default=_faced_DEFAULT_PKL_ROOT)
    parser.add_argument('--metadata-file', type=Path, default=_faced_DEFAULT_METADATA_FILE, help='FACED trial order and label metadata JSON file.')
    parser.add_argument('--output-root', type=Path, default=_faced_DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--validation-root', type=Path, default=_faced_DEFAULT_VALIDATION_ROOT)
    parser.add_argument('--dataset', default=_faced_DEFAULT_DATASET)
    parser.add_argument('--trial-count', type=int, default=_faced_DEFAULT_TRIAL_COUNT)
    parser.add_argument('--subjects', nargs='*', help='Optional pkl subject ids, for example 0 4 or sub000 sub004.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing EDF/NPY/JSON outputs. Existing files are skipped by default.')
    parser.add_argument('--write-validation', action='store_true', help='Write trial validation JSON files. By default only EDF and NPY files are written.')
    parser.add_argument('--no-progress', action='store_true', help='Disable the per-subject progress bar.')
    return parser.parse_args()

def _faced_main() -> None:
    args = _faced_parse_args()
    started = time.time()
    subjects = _faced_list_pkl_subject_indices(args.pkl_root, max_subjects=_faced_MAX_SUBJECTS)
    if args.subjects:
        requested = [_faced_parse_subject_index(subject) for subject in args.subjects]
        available = set(subjects)
        subjects = [subject for subject in requested if subject in available]
        missing = [subject for subject in requested if subject not in available]
        if missing:
            raise SystemExit(f'Requested subjects not found: {missing}')
    if not subjects:
        raise SystemExit(f'No FACED pkl files found under {args.pkl_root}')
    failures: List[Tuple[str, str]] = []
    successes = 0
    subject_iter: Iterable[int] = subjects
    if not args.no_progress:
        subject_iter = _faced_tqdm_iter(subjects, desc=f'Processing {args.dataset}', unit='subject')
    for subject_index in subject_iter:
        try:
            _faced_process_subject(args, subject_index)
            successes += 1
        except Exception:
            error = traceback.format_exc()
            subject_name = f'sub{subject_index:03d}'
            failures.append((subject_name, error))
            print(f'FAILED {subject_name}; continuing. See failure log at end.')
    failed_log = args.output_root / f'{args.dataset}_failed.log'
    if failures:
        _faced_write_failed_log(failed_log, failures)
        print(f'WROTE failure log: {failed_log}')
    elif failed_log.exists() and args.overwrite:
        failed_log.unlink()
    elapsed = time.time() - started
    print(f'\nSummary: total={len(subjects)} success={successes} failed={len(failures)} elapsed_sec={elapsed:.2f}')
    if failures:
        raise SystemExit(1)

# RefedPreprocessor implementation copied from data/preprocess_refed.py.
import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
_refed_DEFAULT_SOURCE_ROOT = DEFAULT_DATASET_ROOT / 'REFED'
_refed_DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_refed_DEFAULT_VALIDATION_ROOT = DEFAULT_VALIDATION_ROOT
_refed_DEFAULT_DATASET = 'REFED'
_refed_DEFAULT_MAX_SUBJECTS = 5
_refed_DEFAULT_SFREQ = 1000.0
_refed_DEFAULT_VIDEO_COUNT = 15
_refed_LABEL_CENTER = 128.0
_refed_DYNAMIC_LABEL_SCALE = 4.0 / 127.0
_refed_SAM_LABEL_CENTER = 5.0
_refed_VIDEO_TARGETED_EMOTION = {1: 0, 2: -1, 3: -1, 4: 1, 5: 1, 6: -1, 7: 1, 8: 1, 9: -1, 10: 0, 11: -1, 12: 0, 13: -1, 14: 1, 15: 1}

def _refed_require_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {module_name}. Install it before running this script.') from exc

def _refed_subject_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return (int(path.name), path.name)
    except ValueError:
        return (10 ** 9, path.name)

def _refed_subject_id_key(value: str) -> str:
    try:
        return str(int(value))
    except ValueError:
        return str(value)

def _refed_list_subject_dirs(source_root: Path) -> List[Path]:
    data_root = source_root / 'data'
    subjects = [item for item in data_root.iterdir() if item.is_dir() and (item / 'EEG_videos.mat').exists()]
    return sorted(subjects, key=_refed_subject_sort_key)[:_refed_DEFAULT_MAX_SUBJECTS]

def _refed_loadmat(path: Path) -> Dict[str, Any]:
    scipy_io = _refed_require_module('scipy.io')
    return scipy_io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

def _refed_load_sam_scores(source_root: Path, subject_id: str, video_count: int) -> Tuple[np.ndarray, np.ndarray]:
    sam_path = source_root / 'SAM_score.csv'
    if not sam_path.exists():
        raise FileNotFoundError(f'Missing REFED SAM score file: {sam_path}')
    with sam_path.open('r', encoding='utf-8-sig', newline='') as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if str(row.get('sub_id', '')).strip() != str(subject_id):
                continue
            valence = []
            arousal = []
            for video_id in range(1, video_count + 1):
                valence.append(float(row[f'Video_{video_id}_Valence']))
                arousal.append(float(row[f'Video_{video_id}_Arousal']))
            return (np.asarray(valence, dtype=np.float32), np.asarray(arousal, dtype=np.float32))
    raise KeyError(f'Subject {subject_id} not found in {sam_path}')

def _refed_read_channel_names(source_root: Path) -> List[str]:
    channels_path = source_root / 'EEG_channels.csv'
    if not channels_path.exists():
        return [f'EEG{idx:03d}' for idx in range(1, 65)]
    with channels_path.open('r', encoding='utf-8-sig', newline='') as file_obj:
        reader = csv.DictReader(file_obj)
        names = [row['ch_name'].strip() for row in reader if row.get('ch_name')]
    if not names:
        raise ValueError(f'No channel names found in {channels_path}')
    return names

def _refed_video_key(video_id: int) -> str:
    return f'video_{video_id}'

def _refed_scale_dynamic_label(raw_label: np.ndarray) -> np.ndarray:
    """Map REFED dynamic labels from raw 0-255 coordinates to the 1-9 SAM scale."""
    return (raw_label.astype(np.float32) - _refed_LABEL_CENTER) * _refed_DYNAMIC_LABEL_SCALE + _refed_SAM_LABEL_CENTER

def _refed_load_subject_eeg_and_labels(source_root: Path, subject_id: str, video_count: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    eeg_mat = _refed_loadmat(source_root / 'data' / subject_id / 'EEG_videos.mat')
    label_mat = _refed_loadmat(source_root / 'annotations' / f'{subject_id}_label.mat')
    eeg_trials: List[np.ndarray] = []
    valence_trials: List[np.ndarray] = []
    arousal_trials: List[np.ndarray] = []
    for vid in range(1, video_count + 1):
        key = _refed_video_key(vid)
        if key not in eeg_mat:
            raise KeyError(f'{key} missing from EEG_videos.mat for subject {subject_id}')
        if key not in label_mat:
            raise KeyError(f'{key} missing from {subject_id}_label.mat')
        eeg = np.asarray(eeg_mat[key], dtype=np.float64)
        labels = np.asarray(label_mat[key])
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError(f'{subject_id} {key} labels shape {labels.shape}; expected (T, 2)')
        if eeg.ndim != 2:
            raise ValueError(f'{subject_id} {key} EEG shape {eeg.shape}; expected (channels, samples)')
        expected_samples = labels.shape[0] * int(_refed_DEFAULT_SFREQ)
        if eeg.shape[1] != expected_samples:
            raise ValueError(f'{subject_id} {key}: EEG samples={eeg.shape[1]} but labels imply {expected_samples} samples ({labels.shape[0]} label points at {_refed_DEFAULT_SFREQ:g}Hz)')
        valence_trials.append(_refed_scale_dynamic_label(labels[:, 0]))
        arousal_trials.append(_refed_scale_dynamic_label(labels[:, 1]))
        eeg_trials.append(eeg)
    return (eeg_trials, valence_trials, arousal_trials)

def _refed_load_targeted_emotion_labels(video_count: int) -> np.ndarray:
    missing = [video_id for video_id in range(1, video_count + 1) if video_id not in _refed_VIDEO_TARGETED_EMOTION]
    if missing:
        raise ValueError(f'Missing REFED targeted emotion mapping for videos: {missing}')
    return np.asarray([_refed_VIDEO_TARGETED_EMOTION[video_id] for video_id in range(1, video_count + 1)], dtype=np.int8)

def _refed_save_dynamic_label_array(path: Path, trials: List[np.ndarray], overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        print(f'SKIP existing label: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(trials), dtype=object)
    for idx, trial in enumerate(trials):
        arr[idx] = trial
    np.save(path, arr, allow_pickle=True)
    print(f'WROTE label: {path}')

def _refed_build_raw(eeg_trials: List[np.ndarray], ch_names: List[str], sfreq: float) -> Any:
    mne = _refed_require_module('mne')
    data = np.concatenate(eeg_trials, axis=1)
    if data.shape[0] != len(ch_names):
        raise ValueError(f'Channel count mismatch: EEG has {data.shape[0]}, CSV has {len(ch_names)}')
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(data, info, first_samp=0, verbose='ERROR')
    raw.set_annotations(None)
    return raw

def _refed_export_raw_to_edf(output_edf: Path, raw: Any, overwrite: bool) -> None:
    _refed_require_module('edfio')
    mne = _refed_require_module('mne')
    output_edf.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(output_edf), raw, fmt='edf', physical_range='channelwise', overwrite=overwrite)

def _refed_build_validate_json(subject_id: str, output_edf: Path, eeg_trials: List[np.ndarray], sam_valence: np.ndarray, sam_arousal: np.ndarray, valence_dynamic_trials: List[np.ndarray], arousal_dynamic_trials: List[np.ndarray], emotion: np.ndarray, sfreq: float) -> Dict[str, Any]:
    trials = []
    start_sample = 0
    for idx, eeg in enumerate(eeg_trials):
        trial_id = idx + 1
        n_samples = int(eeg.shape[1])
        end_sample = start_sample + n_samples
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq
        valence_dynamic = valence_dynamic_trials[idx]
        arousal_dynamic = arousal_dynamic_trials[idx]
        trials.append({'trial_id': trial_id, 'video_id': trial_id, 'video_key': _refed_video_key(trial_id), 'start_sample': start_sample, 'end_sample': end_sample, 'n_samples': n_samples, 'start_sec': start_sec, 'end_sec': end_sec, 'start_min': start_sec / 60.0, 'end_min': end_sec / 60.0, 'duration_sec': n_samples / sfreq, 'label_points': 1, 'dynamic_label_points': int(valence_dynamic.shape[0]), 'valence_mean': float(sam_valence[idx]), 'arousal_mean': float(sam_arousal[idx]), 'valence_dynamic_mean': float(np.mean(valence_dynamic)), 'arousal_dynamic_mean': float(np.mean(arousal_dynamic)), 'emotion': int(emotion[idx])})
        start_sample = end_sample
    return {'dataset': _refed_DEFAULT_DATASET, 'subject': subject_id, 'edf': output_edf.name, 'sfreq': sfreq, 'n_channels': int(eeg_trials[0].shape[0]), 'n_trials': len(eeg_trials), 'label_type': 'static_sam_valence_arousal_with_dynamic_sequences', 'label_center': _refed_LABEL_CENTER, 'emotion_rule': 'targeted video emotion mapping: Neutral/MVMA -> 0; Sad/Fear or LV* -> -1; Happy/Relax or HV* -> 1', 'trials': trials}

def _refed_process_subject(args: argparse.Namespace, subject_dir: Path, ch_names: List[str]) -> None:
    subject_id = subject_dir.name
    output_id = f'{int(subject_id):02d}'
    eeg_trials, valence_dynamic_trials, arousal_dynamic_trials = _refed_load_subject_eeg_and_labels(source_root=args.source_root, subject_id=subject_id, video_count=args.video_count)
    sam_valence, sam_arousal = _refed_load_sam_scores(source_root=args.source_root, subject_id=subject_id, video_count=args.video_count)
    emotion = _refed_load_targeted_emotion_labels(args.video_count)
    raw = _refed_build_raw(eeg_trials=eeg_trials, ch_names=ch_names, sfreq=args.sfreq)
    output_stem = args.output_root / f'{args.dataset}_{output_id}'
    output_edf = output_stem.with_suffix('.edf')
    if output_edf.exists() and (not args.overwrite):
        print(f'SKIP existing EDF: {output_edf}')
    else:
        _refed_export_raw_to_edf(output_edf, raw, overwrite=args.overwrite)
        print(f'WROTE EDF: {output_edf}')
    for label_name, values in (('valence', sam_valence), ('arousal', sam_arousal)):
        label_path = Path(f'{output_stem}_{label_name}.npy')
        if label_path.exists() and (not args.overwrite):
            print(f'SKIP existing label: {label_path}')
        else:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(label_path, np.asarray(values, dtype=np.float32))
            print(f'WROTE label: {label_path}')
    _refed_save_dynamic_label_array(Path(f'{output_stem}_valence_dynamic.npy'), valence_dynamic_trials, overwrite=args.overwrite)
    _refed_save_dynamic_label_array(Path(f'{output_stem}_arousal_dynamic.npy'), arousal_dynamic_trials, overwrite=args.overwrite)
    emotion_path = Path(f'{output_stem}_emotion.npy')
    if emotion_path.exists() and (not args.overwrite):
        print(f'SKIP existing label: {emotion_path}')
    else:
        emotion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emotion_path, emotion)
        print(f'WROTE label: {emotion_path}')
    if args.write_validation:
        validate = _refed_build_validate_json(subject_id=output_id, output_edf=output_edf, eeg_trials=eeg_trials, sam_valence=sam_valence, sam_arousal=sam_arousal, valence_dynamic_trials=valence_dynamic_trials, arousal_dynamic_trials=arousal_dynamic_trials, emotion=emotion, sfreq=args.sfreq)
        validate_dir = args.validation_root / args.dataset
        validate_dir.mkdir(parents=True, exist_ok=True)
        validate_path = validate_dir / f'{args.dataset}_{output_id}_validation.json'
        if validate_path.exists() and (not args.overwrite):
            print(f'SKIP existing validation JSON: {validate_path}')
        else:
            validate_path.write_text(json.dumps(validate, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'WROTE validation JSON: {validate_path}')
    print(f'Processed subject {subject_id}: {len(ch_names)} channels, {raw.n_times} samples, {raw.n_times / args.sfreq:.3f}s -> {output_edf}')

def _refed_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess REFED into EDF + dynamic label npy files.')
    parser.add_argument('--source-root', type=Path, default=_refed_DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=_refed_DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--validation-root', type=Path, default=_refed_DEFAULT_VALIDATION_ROOT)
    parser.add_argument('--dataset', default=_refed_DEFAULT_DATASET)
    parser.add_argument('--sfreq', type=float, default=_refed_DEFAULT_SFREQ)
    parser.add_argument('--video-count', type=int, default=_refed_DEFAULT_VIDEO_COUNT)
    parser.add_argument('--subjects', nargs='*', help='Optional subject ids, for example 1 2 5.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing EDF/NPY/JSON outputs.')
    parser.add_argument('--write-validation', action='store_true', help='Write trial validation JSON files.')
    return parser.parse_args()

def _refed_main() -> None:
    args = _refed_parse_args()
    subjects = _refed_list_subject_dirs(args.source_root)
    if args.subjects:
        wanted = {_refed_subject_id_key(subject) for subject in args.subjects}
        subjects = [subject for subject in subjects if _refed_subject_id_key(subject.name) in wanted]
        missing = sorted(wanted - {_refed_subject_id_key(subject.name) for subject in subjects})
        if missing:
            raise SystemExit(f'Requested subjects not found: {missing}')
    ch_names = _refed_read_channel_names(args.source_root)
    for subject_dir in subjects:
        _refed_process_subject(args, subject_dir, ch_names)

# CogBciPreprocessor implementation copied from data/preprocess_cog_bci.py.
import argparse
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
_cog_bci_DEFAULT_DATASET = 'COG-BCI'
_cog_bci_DEFAULT_SOURCE_ROOT = DEFAULT_DATASET_ROOT / 'COG-BCI'
_cog_bci_DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_cog_bci_DEFAULT_VALIDATION_ROOT = DEFAULT_VALIDATION_ROOT
_cog_bci_DEFAULT_SESSION = 'ses-S1'
_cog_bci_DEFAULT_EPOCH_TMIN_SEC = 0.0
_cog_bci_DEFAULT_EPOCH_DURATION_SEC = 2.0
_cog_bci_TASKS = ({'task': 'zeroBACK', 'behavior_file': '0-Back.mat', 'eeg_file': 'zeroBACK.set', 'condition': 0, 'trial_event_codes': {'6021': 'non_target', '6022': 'target'}}, {'task': 'oneBACK', 'behavior_file': '1-Back.mat', 'eeg_file': 'oneBACK.set', 'condition': 1, 'trial_event_codes': {'6121': 'non_target', '6122': 'target'}}, {'task': 'twoBACK', 'behavior_file': '2-Back.mat', 'eeg_file': 'twoBACK.set', 'condition': 2, 'trial_event_codes': {'6221': 'non_target', '6222': 'target'}})
_cog_bci_LABEL_DTYPE = np.dtype([('trial_id', '<i4'), ('task_trial_id', '<i4'), ('task_index', 'i1'), ('condition', 'i1'), ('task', 'U8'), ('event_code', 'U8'), ('event_type', 'U16'), ('block', '<i2'), ('behavior_class', '<i2'), ('target', 'i1'), ('rt_ms', '<f4'), ('has_response', 'i1'), ('correct', 'i1'), ('miss', 'i1'), ('error', 'i1'), ('mistake', 'i1'), ('outlier', 'i1')])
_cog_bci_MI_NUMPY = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 9: '<f8', 12: '<i8', 13: '<u8'}
_cog_bci_MX_CLASS = {1: 'cell', 2: 'struct', 3: 'object', 4: 'char', 6: 'double', 7: 'single', 8: 'int8', 9: 'uint8', 10: 'int16', 11: 'uint16', 12: 'int32', 13: 'uint32', 14: 'int64', 15: 'uint64'}

@dataclass(frozen=True)
class _cog_bci_SubjectOutputs:
    edf: Path
    label_npy: Path
    validation_json: Path

    def all_paths(self, include_validation: bool) -> List[Path]:
        paths = [self.edf, self.label_npy]
        if include_validation:
            paths.append(self.validation_json)
        return paths

def _cog_bci_require_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {module_name}. Install it before running this script.') from exc

def _cog_bci_subject_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search('(\\d+)$', path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10 ** 9, path.name)

def _cog_bci_subject_id_from_dir(subject_dir: Path) -> str:
    match = re.search('(\\d+)$', subject_dir.name)
    if not match:
        raise ValueError(f'Could not parse subject id from {subject_dir}')
    return f'{int(match.group(1)):02d}'

def _cog_bci_list_subject_dirs(source_root: Path, session: str) -> List[Path]:
    subjects = []
    for item in source_root.iterdir():
        if not item.is_dir() or not item.name.startswith('sub-'):
            continue
        session_dir = item / session
        if not session_dir.exists():
            continue
        if all(((session_dir / 'eeg' / task['eeg_file']).exists() for task in _cog_bci_TASKS)) and all(((session_dir / 'behavioral' / task['behavior_file']).exists() for task in _cog_bci_TASKS)):
            subjects.append(item)
    return sorted(subjects, key=_cog_bci_subject_sort_key)

def _cog_bci_select_subjects(subject_dirs: Sequence[Path], subjects: Optional[Sequence[str]]) -> List[Path]:
    if not subjects:
        return list(subject_dirs)
    wanted = {f'{int(s):02d}' for s in subjects}
    selected = [path for path in subject_dirs if _cog_bci_subject_id_from_dir(path) in wanted]
    missing = wanted - {_cog_bci_subject_id_from_dir(path) for path in selected}
    if missing:
        raise FileNotFoundError(f'Missing requested subjects: {sorted(missing)}')
    return selected

def _cog_bci_pad8(offset: int) -> int:
    return offset + (8 - offset % 8) % 8

def _cog_bci_read_tag(buf: bytes, offset: int) -> Tuple[int, int, int, int]:
    raw = int(np.frombuffer(buf[offset:offset + 4], dtype='<u4')[0])
    small_type = raw & 65535
    small_nbytes = raw >> 16
    if small_type and small_nbytes and (small_nbytes <= 4):
        return (small_type, small_nbytes, offset + 4, offset + 8)
    dtype, nbytes = np.frombuffer(buf[offset:offset + 8], dtype='<u4')
    start = offset + 8
    return (int(dtype), int(nbytes), start, start + int(nbytes))

def _cog_bci_read_numeric(buf: bytes, dtype: int, start: int, stop: int) -> np.ndarray:
    if dtype not in _cog_bci_MI_NUMPY:
        raise ValueError(f'Unsupported MATLAB numeric dtype {dtype}')
    return np.frombuffer(buf[start:stop], dtype=np.dtype(_cog_bci_MI_NUMPY[dtype])).copy()

def _cog_bci_parse_matrix(buf: bytes, offset: int, stop: int) -> Any:
    if stop <= offset:
        return {'kind': 'empty'}
    dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
    flags = buf[start:end]
    offset = _cog_bci_pad8(end)
    class_id = flags[0] if flags else None
    kind = _cog_bci_MX_CLASS.get(class_id, str(class_id))
    dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
    dims = tuple(_cog_bci_read_numeric(buf, dtype, start, end).astype(int).tolist()) if nbytes else ()
    offset = _cog_bci_pad8(end)
    dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
    name = buf[start:end].decode('latin1', 'replace').rstrip('\x00')
    offset = _cog_bci_pad8(end)
    numeric_kinds = {'double', 'single', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'}
    if kind in numeric_kinds:
        if offset >= stop:
            return {'kind': kind, 'dims': dims, 'name': name, 'data': np.array([])}
        dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
        arr = _cog_bci_read_numeric(buf, dtype, start, end) if nbytes else np.array([])
        if dims and arr.size == math.prod(dims):
            arr = arr.reshape(dims, order='F')
        return {'kind': kind, 'dims': dims, 'name': name, 'data': arr}
    if kind == 'char':
        if offset >= stop:
            return ''
        dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
        raw = buf[start:end]
        if dtype in {1, 2, 16}:
            encoding = 'utf-8'
        elif dtype == 17:
            encoding = 'utf-16le'
        elif dtype == 18:
            encoding = 'utf-32le'
        else:
            encoding = 'latin1'
        return raw.decode(encoding, 'replace').rstrip('\x00')
    elems = []
    while offset < stop:
        dtype, nbytes, start, end = _cog_bci_read_tag(buf, offset)
        if dtype == 14:
            elems.append({'kind': 'empty'} if nbytes == 0 else _cog_bci_parse_matrix(buf, start, end))
        elif dtype in {1, 2, 16}:
            elems.append(buf[start:end].decode('latin1', 'replace').rstrip('\x00'))
        elif dtype in _cog_bci_MI_NUMPY:
            elems.append(_cog_bci_read_numeric(buf, dtype, start, end))
        else:
            elems.append({'kind': 'unknown', 'dtype': dtype, 'nbytes': nbytes})
        offset = _cog_bci_pad8(end)
    return {'kind': kind, 'dims': dims, 'name': name, 'elems': elems}

def _cog_bci_walk(node: Any) -> Iterable[Any]:
    yield node
    if isinstance(node, dict):
        for elem in node.get('elems', []):
            yield from _cog_bci_walk(elem)

def _cog_bci_load_nback_table(path: Path) -> Dict[str, np.ndarray]:
    scipy_io = _cog_bci_require_module('scipy.io')
    mat = scipy_io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if '__function_workspace__' not in mat:
        raise KeyError(f'{path} does not contain __function_workspace__')
    workspace = np.asarray(mat['__function_workspace__'], dtype=np.uint8).reshape(-1).tobytes()
    dtype, nbytes, start, stop = _cog_bci_read_tag(workspace, 8)
    if dtype != 14:
        raise ValueError(f'{path}: expected inner miMATRIX tag at offset 8, found dtype={dtype}')
    root = _cog_bci_parse_matrix(workspace, start, stop)
    column_names = None
    column_arrays = None
    for node in _cog_bci_walk(root):
        if not isinstance(node, dict):
            continue
        if node.get('kind') != 'cell' or node.get('dims') != (1, 13):
            continue
        elems = node.get('elems', [])
        if len(elems) == 13 and all((isinstance(elem, str) for elem in elems)):
            column_names = elems
        if len(elems) == 13 and all((isinstance(elem, dict) and 'data' in elem for elem in elems)):
            column_arrays = [elem['data'].reshape(-1, order='F') for elem in elems]
    if column_names is None or column_arrays is None:
        raise RuntimeError(f'Could not parse MATLAB table from {path}')
    table = {name: np.asarray(arr) for name, arr in zip(column_names, column_arrays)}
    row_counts = {len(arr) for arr in table.values()}
    if len(row_counts) != 1:
        raise ValueError(f'{path}: inconsistent behavioral column lengths: {row_counts}')
    return table

def _cog_bci_float_or_none(value: float) -> Optional[float]:
    value = float(value)
    if math.isnan(value):
        return None
    return value

def _cog_bci_int_value(value: float) -> int:
    return int(round(float(value)))

def _cog_bci_load_raw_eeglab(path: Path) -> Any:
    mne = _cog_bci_require_module('mne')
    return mne.io.read_raw_eeglab(str(path), preload=True, verbose='ERROR')

def _cog_bci_extract_trial_events(raw: Any, trial_event_codes: Mapping[str, str]) -> List[Dict[str, Any]]:
    events = []
    sfreq = float(raw.info['sfreq'])
    for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        event_code = str(description)
        if event_code not in trial_event_codes:
            continue
        start_sample = int(round(float(onset) * sfreq))
        events.append({'event_code': event_code, 'event_type': trial_event_codes[event_code], 'onset_sec': float(onset), 'onset_sample': start_sample, 'annotation_duration_sec': float(duration)})
    return sorted(events, key=lambda item: item['onset_sec'])

def _cog_bci_check_behavior_event_alignment(behavior: Mapping[str, np.ndarray], events: Sequence[Mapping[str, Any]], task: str) -> None:
    n_behavior = len(next(iter(behavior.values())))
    if len(events) != n_behavior:
        raise ValueError(f'{task}: EEG trial events={len(events)} but behavioral rows={n_behavior}')
    hittrials = np.asarray(behavior['hittrials'])
    for idx, event in enumerate(events):
        target_from_behavior = _cog_bci_int_value(hittrials[idx]) == 1
        target_from_event = event['event_type'] == 'target'
        if target_from_behavior != target_from_event:
            raise ValueError(f"{task}: event/behavior target mismatch at row {idx + 1}: event={event['event_code']}({event['event_type']}), hittrials={hittrials[idx]}")

def _cog_bci_make_subject_outputs(output_root: Path, validation_root: Path, subject_id: str) -> _cog_bci_SubjectOutputs:
    stem = f'{_cog_bci_DEFAULT_DATASET}_{subject_id}'
    return _cog_bci_SubjectOutputs(edf=output_root / f'{stem}.edf', label_npy=output_root / f'{stem}.npy', validation_json=validation_root / _cog_bci_DEFAULT_DATASET / f'{stem}_validation.json')

def _cog_bci_assert_outputs_available(outputs: _cog_bci_SubjectOutputs, overwrite: bool, include_validation: bool) -> None:
    if overwrite:
        return
    existing = [path for path in outputs.all_paths(include_validation) if path.exists()]
    if existing:
        joined = '\n  '.join((str(path) for path in existing))
        raise FileExistsError(f'Refusing to overwrite existing outputs:\n  {joined}')

def _cog_bci_build_label_row(trial_id: int, task_trial_id: int, task_index: int, task_spec: Mapping[str, Any], event: Mapping[str, Any], behavior: Mapping[str, np.ndarray], row_idx: int) -> np.void:
    row = np.zeros((), dtype=_cog_bci_LABEL_DTYPE)
    rt = float(behavior['rt'][row_idx])
    target = _cog_bci_int_value(behavior['hittrials'][row_idx])
    row['trial_id'] = trial_id
    row['task_trial_id'] = task_trial_id
    row['task_index'] = task_index
    row['condition'] = int(task_spec['condition'])
    row['task'] = str(task_spec['task'])
    row['event_code'] = str(event['event_code'])
    row['event_type'] = str(event['event_type'])
    row['block'] = _cog_bci_int_value(behavior['block'][row_idx])
    row['behavior_class'] = _cog_bci_int_value(behavior['class'][row_idx])
    row['target'] = target
    row['rt_ms'] = np.nan if math.isnan(rt) else rt
    row['has_response'] = 0 if math.isnan(rt) else 1
    row['correct'] = _cog_bci_int_value(behavior['correct'][row_idx])
    row['miss'] = _cog_bci_int_value(behavior['miss'][row_idx])
    row['error'] = _cog_bci_int_value(behavior['error'][row_idx])
    row['mistake'] = _cog_bci_int_value(behavior['mistake'][row_idx])
    row['outlier'] = _cog_bci_int_value(behavior['outlier'][row_idx])
    return row

def _cog_bci_label_row_to_json(row: np.void) -> Dict[str, Any]:
    rt = float(row['rt_ms'])
    return {'condition': int(row['condition']), 'task': str(row['task']), 'event_code': str(row['event_code']), 'event_type': str(row['event_type']), 'block': int(row['block']), 'behavior_class': int(row['behavior_class']), 'target': int(row['target']), 'rt_ms': None if math.isnan(rt) else rt, 'has_response': int(row['has_response']), 'correct': int(row['correct']), 'miss': int(row['miss']), 'error': int(row['error']), 'mistake': int(row['mistake']), 'outlier': int(row['outlier'])}

def _cog_bci_build_concatenated_raw(raws: Sequence[Any]) -> Any:
    mne = _cog_bci_require_module('mne')
    first = raws[0]
    sfreq = float(first.info['sfreq'])
    ch_names = list(first.ch_names)
    ch_types = first.get_channel_types()
    for idx, raw in enumerate(raws[1:], start=2):
        if float(raw.info['sfreq']) != sfreq:
            raise ValueError(f"Task {idx} has sfreq={raw.info['sfreq']}, expected {sfreq}")
        if list(raw.ch_names) != ch_names:
            raise ValueError(f'Task {idx} channel names/order differ from the first task')
        if raw.get_channel_types() != ch_types:
            raise ValueError(f'Task {idx} channel types differ from the first task')
    data = np.concatenate([raw.get_data() for raw in raws], axis=1)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    merged = mne.io.RawArray(data, info, first_samp=0, verbose='ERROR')
    merged.set_annotations(None)
    return merged

def _cog_bci_export_raw_to_edf(output_edf: Path, raw: Any, overwrite: bool) -> None:
    _cog_bci_require_module('edfio')
    mne = _cog_bci_require_module('mne')
    output_edf.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(output_edf), raw, fmt='edf', physical_range='channelwise', overwrite=overwrite)

def _cog_bci_build_subject(subject_dir: Path, session: str, output_root: Path, validation_root: Path, epoch_tmin_sec: float, epoch_duration_sec: float, overwrite: bool, dry_run: bool, write_validation: bool) -> None:
    subject_id = _cog_bci_subject_id_from_dir(subject_dir)
    session_dir = subject_dir / session
    eeg_dir = session_dir / 'eeg'
    behavior_dir = session_dir / 'behavioral'
    outputs = _cog_bci_make_subject_outputs(output_root, validation_root, subject_id)
    _cog_bci_assert_outputs_available(outputs, overwrite=overwrite, include_validation=write_validation)
    raws = []
    labels: List[np.void] = []
    trials: List[Dict[str, Any]] = []
    cumulative_samples = 0
    trial_id = 1
    for task_index, task_spec in enumerate(_cog_bci_TASKS):
        raw = _cog_bci_load_raw_eeglab(eeg_dir / str(task_spec['eeg_file']))
        behavior = _cog_bci_load_nback_table(behavior_dir / str(task_spec['behavior_file']))
        events = _cog_bci_extract_trial_events(raw, task_spec['trial_event_codes'])
        _cog_bci_check_behavior_event_alignment(behavior, events, str(task_spec['task']))
        sfreq = float(raw.info['sfreq'])
        epoch_samples = int(round(epoch_duration_sec * sfreq))
        tmin_samples = int(round(epoch_tmin_sec * sfreq))
        for row_idx, event in enumerate(events):
            task_trial_id = row_idx + 1
            label_row = _cog_bci_build_label_row(trial_id=trial_id, task_trial_id=task_trial_id, task_index=task_index, task_spec=task_spec, event=event, behavior=behavior, row_idx=row_idx)
            start_sample = cumulative_samples + int(event['onset_sample']) + tmin_samples
            end_sample = start_sample + epoch_samples
            if start_sample < cumulative_samples:
                raise ValueError(f"{subject_id} {task_spec['task']} trial {task_trial_id}: negative epoch start")
            if end_sample > cumulative_samples + int(raw.n_times):
                raise ValueError(f"{subject_id} {task_spec['task']} trial {task_trial_id}: epoch exceeds task data range")
            start_sec = start_sample / sfreq
            end_sec = end_sample / sfreq
            trial_json = {'trial_id': trial_id, 'task_trial_id': task_trial_id, 'task_index': task_index, 'source_subject': subject_dir.name, 'source_session': session, 'source_behavior_file': str(task_spec['behavior_file']), 'source_eeg_file': str(task_spec['eeg_file']), 'source_event_onset_sec': float(event['onset_sec']), 'source_event_onset_sample': int(event['onset_sample']), 'start_sample': int(start_sample), 'end_sample': int(end_sample), 'n_samples': int(epoch_samples), 'start_sec': float(start_sec), 'end_sec': float(end_sec), 'duration_sec': float(epoch_duration_sec), **_cog_bci_label_row_to_json(label_row)}
            labels.append(label_row)
            trials.append(trial_json)
            trial_id += 1
        raws.append(raw)
        cumulative_samples += int(raw.n_times)
    merged = _cog_bci_build_concatenated_raw(raws)
    labels_arr = np.asarray(labels, dtype=_cog_bci_LABEL_DTYPE)
    validation = {'dataset': _cog_bci_DEFAULT_DATASET, 'subject': subject_id, 'source_subject': subject_dir.name, 'source_session': session, 'edf': outputs.edf.name, 'label_npy': outputs.label_npy.name, 'sfreq': float(merged.info['sfreq']), 'n_channels': len(merged.ch_names), 'n_trials': len(trials), 'label_type': 'nback_behavioral_trial_labels', 'epoch_tmin_sec': float(epoch_tmin_sec), 'epoch_duration_sec': float(epoch_duration_sec), 'tasks': [{'task': str(task['task']), 'condition': int(task['condition']), 'behavior_file': str(task['behavior_file']), 'eeg_file': str(task['eeg_file']), 'trial_event_codes': dict(task['trial_event_codes'])} for task in _cog_bci_TASKS], 'trials': trials}
    print(f"{_cog_bci_DEFAULT_DATASET}_{subject_id}: channels={len(merged.ch_names)} sfreq={merged.info['sfreq']:.1f}Hz samples={merged.n_times} trials={len(trials)}")
    if dry_run:
        print(f'DRY-RUN would write: {outputs.edf}')
        print(f'DRY-RUN would write: {outputs.label_npy}')
        if write_validation:
            print(f'DRY-RUN would write: {outputs.validation_json}')
        return
    output_root.mkdir(parents=True, exist_ok=True)
    _cog_bci_export_raw_to_edf(outputs.edf, merged, overwrite=overwrite)
    np.save(outputs.label_npy, labels_arr, allow_pickle=False)
    if write_validation:
        outputs.validation_json.parent.mkdir(parents=True, exist_ok=True)
        with outputs.validation_json.open('w', encoding='utf-8') as file_obj:
            json.dump(validation, file_obj, ensure_ascii=False, indent=2)
    print(f'WROTE {outputs.edf}')
    print(f'WROTE {outputs.label_npy}')
    if write_validation:
        print(f'WROTE {outputs.validation_json}')

def _cog_bci_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess COG-BCI N-back ses-S1 data for NeuroBench.')
    parser.add_argument('--source-root', type=Path, default=_cog_bci_DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=_cog_bci_DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--validation-root', type=Path, default=_cog_bci_DEFAULT_VALIDATION_ROOT)
    parser.add_argument('--session', default=_cog_bci_DEFAULT_SESSION)
    parser.add_argument('--subjects', nargs='*', help='Optional subject ids, e.g. 1 2 03')
    parser.add_argument('--max-subjects', type=int, default=None)
    parser.add_argument('--epoch-tmin-sec', type=float, default=_cog_bci_DEFAULT_EPOCH_TMIN_SEC)
    parser.add_argument('--epoch-duration-sec', type=float, default=_cog_bci_DEFAULT_EPOCH_DURATION_SEC)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write-validation', action='store_true', help='Write validation JSON files.')
    return parser.parse_args()

def _cog_bci_main() -> None:
    args = _cog_bci_parse_args()
    subjects = _cog_bci_select_subjects(_cog_bci_list_subject_dirs(args.source_root, args.session), args.subjects)
    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]
    if not subjects:
        raise SystemExit(f'No subjects found under {args.source_root} for {args.session}')
    for subject_dir in subjects:
        _cog_bci_build_subject(subject_dir=subject_dir, session=args.session, output_root=args.output_root, validation_root=args.validation_root, epoch_tmin_sec=args.epoch_tmin_sec, epoch_duration_sec=args.epoch_duration_sec, overwrite=args.overwrite, dry_run=args.dry_run, write_validation=args.write_validation)

# MpdDfPreprocessor implementation copied from data/preprocess_mpd_df.py.
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
_mpd_df_DEFAULT_DATASET = 'MPD-DF'
_mpd_df_DEFAULT_SOURCE_ROOT = DEFAULT_DATASET_ROOT / 'MPD-DF'
_mpd_df_DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_mpd_df_DEFAULT_VALIDATION_ROOT = DEFAULT_VALIDATION_ROOT
_mpd_df_DEFAULT_PREPROCESS_ROOT = SCRIPT_DIR
_mpd_df_RECOMMENDED_SUBJECTS = ('01', '02', '03', '04', '06')
_mpd_df_EEG_SUBDIR = 'EEG'
_mpd_df_PSG_SUBDIR = 'PSG'
_mpd_df_ANNOTATION_SUBDIR = 'Annotation'
_mpd_df_FATIGUE_INTERVAL_DTYPE = np.dtype([('segment_id', '<i4'), ('start_sec', '<f8'), ('end_sec', '<f8'), ('start_datetime', 'U19'), ('end_datetime', 'U19'), ('source_label', 'U32'), ('label', '<i2'), ('label_name', 'U32')])
_mpd_df_LABEL_ENCODING: Mapping[str, Tuple[int, str]] = {'0': (0, 'Wakefulness'), '1': (1, 'Fatigue1'), '2': (2, 'Fatigue2'), '3': (3, 'Fatigue3'), '4': (4, 'Fatigue4'), 'Signal Abnormality': (-1, 'Signal Abnormality'), 'Severe Artifacts': (-2, 'Severe Artifacts')}

@dataclass(frozen=True)
class _mpd_df_EdfInfo:
    path: Path
    start: datetime
    n_records: int
    record_duration_sec: float
    n_signals: int
    channels: Tuple[str, ...]
    samples_per_record: Tuple[int, ...]
    header_bytes: int
    record_bytes: int

    @property
    def duration_sec(self) -> float:
        return float(self.n_records * self.record_duration_sec)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_sec)

    @property
    def sfreq_by_channel(self) -> Dict[str, float]:
        return {channel: float(samples / self.record_duration_sec) for channel, samples in zip(self.channels, self.samples_per_record)}

@dataclass(frozen=True)
class _mpd_df_SubjectFiles:
    subject: str
    eeg: Path
    psg: Path
    annotation: Path

@dataclass(frozen=True)
class _mpd_df_SubjectOutputs:
    eeg_edf: Path
    psg_edf: Path
    fatigue_npy: Path
    validation_json: Path

    def all_paths(self, include_validation: bool) -> List[Path]:
        paths = [self.eeg_edf, self.psg_edf, self.fatigue_npy]
        if include_validation:
            paths.append(self.validation_json)
        return paths

def _mpd_df_parse_edf_datetime(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f'{date_text} {time_text}', '%d.%m.%y %H.%M.%S')

def _mpd_df_edf_ascii_int(raw: bytes) -> int:
    return int(raw.decode('ascii', 'replace').strip())

def _mpd_df_edf_ascii_float(raw: bytes) -> float:
    return float(raw.decode('ascii', 'replace').strip())

def _mpd_df_read_edf_info(path: Path) -> _mpd_df_EdfInfo:
    with path.open('rb') as file_obj:
        fixed_header = file_obj.read(256)
        if len(fixed_header) != 256:
            raise ValueError(f'{path} is too small to be an EDF file')
        header_bytes = _mpd_df_edf_ascii_int(fixed_header[184:192])
        n_records = _mpd_df_edf_ascii_int(fixed_header[236:244])
        record_duration = _mpd_df_edf_ascii_float(fixed_header[244:252])
        n_signals = _mpd_df_edf_ascii_int(fixed_header[252:256])
        signal_header = file_obj.read(n_signals * 256)
        if len(signal_header) != n_signals * 256:
            raise ValueError(f'{path} has a truncated signal header')
    start = _mpd_df_parse_edf_datetime(fixed_header[168:176].decode('ascii').strip(), fixed_header[176:184].decode('ascii').strip())
    channels = tuple((signal_header[idx * 16:(idx + 1) * 16].decode('latin1').strip() for idx in range(n_signals)))
    samples_offset = 216 * n_signals
    samples_per_record = tuple((_mpd_df_edf_ascii_int(signal_header[samples_offset + idx * 8:samples_offset + (idx + 1) * 8]) for idx in range(n_signals)))
    record_bytes = int(sum(samples_per_record) * 2)
    expected_header_bytes = 256 + n_signals * 256
    if header_bytes != expected_header_bytes:
        raise ValueError(f'{path}: header_bytes={header_bytes}, expected {expected_header_bytes}')
    if record_bytes <= 0:
        raise ValueError(f'{path}: invalid EDF data record size {record_bytes}')
    return _mpd_df_EdfInfo(path=path, start=start, n_records=n_records, record_duration_sec=record_duration, n_signals=n_signals, channels=channels, samples_per_record=samples_per_record, header_bytes=header_bytes, record_bytes=record_bytes)

def _mpd_df_write_left_justified_ascii(buf: bytearray, start: int, stop: int, value: str) -> None:
    width = stop - start
    encoded = value.encode('ascii')
    if len(encoded) > width:
        raise ValueError(f'Value {value!r} exceeds EDF field width {width}')
    buf[start:stop] = encoded.ljust(width, b' ')

def _mpd_df_crop_edf_by_records(source: Path, output: Path, output_start: datetime, start_offset_sec: float, duration_sec: int, overwrite: bool) -> None:
    info = _mpd_df_read_edf_info(source)
    if not math.isclose(info.record_duration_sec, 1.0, rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'{source}: expected 1-second EDF records for safe record-copy crop, got {info.record_duration_sec}')
    if not math.isclose(start_offset_sec, round(start_offset_sec), rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'{source}: non-integer crop offset {start_offset_sec}')
    start_record = int(round(start_offset_sec / info.record_duration_sec))
    n_records = int(duration_sec / info.record_duration_sec)
    if start_record < 0:
        raise ValueError(f'{source}: negative crop record {start_record}')
    if start_record + n_records > info.n_records:
        raise ValueError(f'{source}: crop {start_record}:{start_record + n_records} exceeds {info.n_records} records')
    if output.exists() and (not overwrite):
        print(f'SKIP existing EDF: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src:
        header = bytearray(src.read(info.header_bytes))
        _mpd_df_write_left_justified_ascii(header, 168, 176, output_start.strftime('%d.%m.%y'))
        _mpd_df_write_left_justified_ascii(header, 176, 184, output_start.strftime('%H.%M.%S'))
        _mpd_df_write_left_justified_ascii(header, 236, 244, str(n_records))
        src.seek(info.header_bytes + start_record * info.record_bytes)
        bytes_to_copy = n_records * info.record_bytes
        data = src.read(bytes_to_copy)
        if len(data) != bytes_to_copy:
            raise ValueError(f'{source}: truncated EDF data while cropping')
    with output.open('wb') as dst:
        dst.write(header)
        dst.write(data)
    print(f'WROTE EDF: {output}')

def _mpd_df_subject_sort_key(text: str) -> Tuple[int, str]:
    match = re.search('(\\d+)', text)
    if match:
        return (int(match.group(1)), text)
    return (10 ** 9, text)

def _mpd_df_subject_files(source_root: Path, subject: str) -> _mpd_df_SubjectFiles:
    return _mpd_df_SubjectFiles(subject=subject, eeg=source_root / _mpd_df_EEG_SUBDIR / f'MPDDF_raw_{subject}_EEG.edf', psg=source_root / _mpd_df_PSG_SUBDIR / f'MPDDF_raw_{subject}_PSG.edf', annotation=source_root / _mpd_df_ANNOTATION_SUBDIR / f'MPDDF_raw_{subject}_Annotation.txt')

def _mpd_df_list_subjects(source_root: Path) -> List[_mpd_df_SubjectFiles]:
    eeg_dir = source_root / _mpd_df_EEG_SUBDIR
    subjects = []
    for eeg_path in eeg_dir.glob('MPDDF_raw_*_EEG.edf'):
        match = re.search('MPDDF_raw_(\\d+)_EEG\\.edf$', eeg_path.name)
        if not match:
            continue
        subject = f'{int(match.group(1)):02d}'
        files = _mpd_df_subject_files(source_root, subject)
        missing = [path for path in (files.eeg, files.psg, files.annotation) if not path.exists()]
        if missing:
            print(f'SKIP incomplete subject {subject}: missing {missing}')
            continue
        subjects.append(files)
    return sorted(subjects, key=lambda item: _mpd_df_subject_sort_key(item.subject))

def _mpd_df_select_subjects(all_subjects: Sequence[_mpd_df_SubjectFiles], wanted: Optional[Sequence[str]]) -> List[_mpd_df_SubjectFiles]:
    if not wanted:
        return list(all_subjects)
    wanted_ids = {f'{int(subject):02d}' for subject in wanted}
    selected = [files for files in all_subjects if files.subject in wanted_ids]
    missing = wanted_ids - {files.subject for files in selected}
    if missing:
        raise FileNotFoundError(f'Requested subjects not found: {sorted(missing)}')
    return selected

def _mpd_df_combine_time_with_date(time_text: str, base: datetime) -> datetime:
    combined = datetime.combine(base.date(), datetime.strptime(time_text, '%H:%M:%S').time())
    if combined < base - timedelta(hours=12):
        combined += timedelta(days=1)
    return combined

def _mpd_df_read_annotation_rows(annotation_path: Path, base: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with annotation_path.open('r', encoding='utf-8-sig', newline='') as file_obj:
        reader = csv.reader(file_obj)
        for row_idx, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f'{annotation_path} row {row_idx}: expected 3 columns, got {row}')
            source_time = row[0].strip()
            segment_index = int(row[1])
            source_label = row[2].strip()
            if source_label not in _mpd_df_LABEL_ENCODING:
                raise ValueError(f'{annotation_path} row {row_idx}: unknown label {source_label!r}')
            start_abs = _mpd_df_combine_time_with_date(source_time, base)
            if rows and start_abs < rows[-1]['source_start_abs']:
                start_abs += timedelta(days=1)
            rows.append({'source_annotation_row': row_idx, 'source_time': source_time, 'source_segment_index': segment_index, 'source_label': source_label, 'source_start_abs': start_abs})
    if not rows:
        raise ValueError(f'{annotation_path} is empty')
    return rows

def _mpd_df_iso(dt: datetime) -> str:
    return dt.isoformat(timespec='seconds')

def _mpd_df_seconds_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds()

def _mpd_df_label_summary(intervals: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for interval in intervals:
        label_name = str(interval['label_name'])
        duration = float(interval['output_end_sec']) - float(interval['output_start_sec'])
        summary[label_name] = summary.get(label_name, 0.0) + duration
    return {key: round(value, 6) for key, value in sorted(summary.items())}

def _mpd_df_build_label_intervals(rows: Sequence[Mapping[str, Any]], eeg_start: datetime, eeg_end: datetime, align_start: datetime, usable_duration_sec: int) -> List[Dict[str, Any]]:
    crop_start = align_start
    crop_end = align_start + timedelta(seconds=usable_duration_sec)
    intervals: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_start = row['source_start_abs']
        source_end = rows[idx + 1]['source_start_abs'] if idx + 1 < len(rows) else eeg_end
        if source_end < source_start:
            raise ValueError(f'Annotation row {idx + 1}: end before start')
        clipped_start = max(source_start, crop_start)
        clipped_end = min(source_end, crop_end)
        if clipped_end <= clipped_start:
            continue
        label, label_name = _mpd_df_LABEL_ENCODING[str(row['source_label'])]
        output_start_sec = _mpd_df_seconds_between(clipped_start, align_start)
        output_end_sec = _mpd_df_seconds_between(clipped_end, align_start)
        segment_id = len(intervals) + 1
        intervals.append({'segment_id': int(segment_id), 'trial_id': int(segment_id), 'source_annotation_row': int(row['source_annotation_row']), 'source_segment_index': int(row['source_segment_index']), 'source_label': str(row['source_label']), 'label': int(label), 'label_name': label_name, 'source_start_time': str(row['source_time']), 'source_end_time': source_end.strftime('%H:%M:%S'), 'source_start_abs': _mpd_df_iso(source_start), 'source_end_abs': _mpd_df_iso(source_end), 'source_start_sec_from_eeg': _mpd_df_seconds_between(source_start, eeg_start), 'source_end_sec_from_eeg': _mpd_df_seconds_between(source_end, eeg_start), 'clipped_start_abs': _mpd_df_iso(clipped_start), 'clipped_end_abs': _mpd_df_iso(clipped_end), 'output_start_sec': float(output_start_sec), 'output_end_sec': float(output_end_sec), 'start_sec': float(output_start_sec), 'end_sec': float(output_end_sec), 'duration_sec': float(output_end_sec - output_start_sec), 'calculation': {'output_start_sec': 'clipped_start_abs - align_start', 'output_end_sec': 'clipped_end_abs - align_start', 'clip_window': '[align_start, align_start + usable_duration_sec)'}})
    return intervals

def _mpd_df_save_fatigue_intervals(path: Path, intervals: Sequence[Mapping[str, Any]], overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        print(f'SKIP existing label: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.empty(len(intervals), dtype=_mpd_df_FATIGUE_INTERVAL_DTYPE)
    for idx, interval in enumerate(intervals):
        array[idx] = (int(interval['segment_id']), float(interval['output_start_sec']), float(interval['output_end_sec']), str(interval['clipped_start_abs']), str(interval['clipped_end_abs']), str(interval['source_label']), int(interval['label']), str(interval['label_name']))
    np.save(path, array)
    print(f'WROTE label: {path}')

def _mpd_df_output_paths(dataset: str, subject: str, output_root: Path, validation_root: Path) -> _mpd_df_SubjectOutputs:
    stem = f'{dataset}_{subject}'
    return _mpd_df_SubjectOutputs(eeg_edf=output_root / f'{stem}.edf', psg_edf=output_root / f'{stem}_PSG.edf', fatigue_npy=output_root / f'{stem}_fatigue.npy', validation_json=validation_root / dataset / f'{stem}_validation.json')

def _mpd_df_build_validation_json(dataset: str, subject: str, files: _mpd_df_SubjectFiles, outputs: _mpd_df_SubjectOutputs, eeg_info: _mpd_df_EdfInfo, psg_info: _mpd_df_EdfInfo, annotation_start: datetime, align_start: datetime, align_end: datetime, usable_duration_sec: int, intervals: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {'dataset': dataset, 'subject': subject, 'source_files': {'eeg': str(files.eeg), 'psg': str(files.psg), 'annotation': str(files.annotation)}, 'outputs': {'edf': outputs.eeg_edf.name, 'eeg_edf': outputs.eeg_edf.name, 'psg_edf': outputs.psg_edf.name, 'fatigue_npy': outputs.fatigue_npy.name}, 'time_base': 'seconds_from_aligned_output_edf_start', 'align_start': _mpd_df_iso(align_start), 'align_end': _mpd_df_iso(align_end), 'usable_duration_sec': int(usable_duration_sec), 'offsets_sec': {'eeg': _mpd_df_seconds_between(align_start, eeg_info.start), 'psg': _mpd_df_seconds_between(align_start, psg_info.start), 'annotation': _mpd_df_seconds_between(align_start, annotation_start)}, 'source_timing': {'eeg_start': _mpd_df_iso(eeg_info.start), 'eeg_end': _mpd_df_iso(eeg_info.end), 'eeg_duration_sec': eeg_info.duration_sec, 'psg_start': _mpd_df_iso(psg_info.start), 'psg_end': _mpd_df_iso(psg_info.end), 'psg_duration_sec': psg_info.duration_sec}, 'eeg': {'sfreq': next(iter(eeg_info.sfreq_by_channel.values())), 'n_channels': eeg_info.n_signals, 'channels': list(eeg_info.channels), 'samples_per_record': list(eeg_info.samples_per_record)}, 'psg': {'n_channels': psg_info.n_signals, 'channels': list(psg_info.channels), 'samples_per_record': list(psg_info.samples_per_record), 'sfreq_by_channel': psg_info.sfreq_by_channel}, 'label_type': 'fatigue_intervals', 'label_file_format': 'structured_npy[(segment_id,start_sec,end_sec,start_datetime,end_datetime,source_label,label,label_name)]', 'label_encoding': {str(value): name for _raw, (value, name) in _mpd_df_LABEL_ENCODING.items()}, 'n_trials': len(intervals), 'n_label_intervals': len(intervals), 'label_duration_summary_sec': _mpd_df_label_summary(intervals), 'trials': list(intervals), 'label_intervals': list(intervals)}

def _mpd_df_save_validation_json(path: Path, validation: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        print(f'SKIP existing validation JSON: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_obj:
        json.dump(validation, file_obj, indent=2, ensure_ascii=False)
        file_obj.write('\n')
    print(f'WROTE validation JSON: {path}')

def _mpd_df_assert_outputs(outputs: _mpd_df_SubjectOutputs, include_validation: bool) -> None:
    missing = [str(path) for path in outputs.all_paths(include_validation) if not path.exists()]
    if missing:
        raise AssertionError(f'Missing expected output files: {missing}')

def _mpd_df_process_subject(args: argparse.Namespace, files: _mpd_df_SubjectFiles) -> _mpd_df_SubjectOutputs:
    outputs = _mpd_df_output_paths(args.dataset, files.subject, args.output_root, args.validation_root)
    eeg_info = _mpd_df_read_edf_info(files.eeg)
    psg_info = _mpd_df_read_edf_info(files.psg)
    annotation_rows = _mpd_df_read_annotation_rows(files.annotation, eeg_info.start)
    annotation_start = annotation_rows[0]['source_start_abs']
    align_start = max(eeg_info.start, psg_info.start, annotation_start)
    raw_align_end = min(eeg_info.end, psg_info.end)
    usable_duration_sec = int(math.floor(_mpd_df_seconds_between(raw_align_end, align_start)))
    if usable_duration_sec <= 0:
        raise ValueError(f'Subject {files.subject}: non-positive aligned duration')
    align_end = align_start + timedelta(seconds=usable_duration_sec)
    eeg_offset_sec = _mpd_df_seconds_between(align_start, eeg_info.start)
    psg_offset_sec = _mpd_df_seconds_between(align_start, psg_info.start)
    intervals = _mpd_df_build_label_intervals(rows=annotation_rows, eeg_start=eeg_info.start, eeg_end=eeg_info.end, align_start=align_start, usable_duration_sec=usable_duration_sec)
    if not intervals:
        raise ValueError(f'Subject {files.subject}: no label intervals after alignment')
    print(f'\nProcessing {args.dataset}_{files.subject}: align_start={_mpd_df_iso(align_start)}, duration={usable_duration_sec}s')
    _mpd_df_crop_edf_by_records(source=files.eeg, output=outputs.eeg_edf, output_start=align_start, start_offset_sec=eeg_offset_sec, duration_sec=usable_duration_sec, overwrite=args.overwrite)
    _mpd_df_crop_edf_by_records(source=files.psg, output=outputs.psg_edf, output_start=align_start, start_offset_sec=psg_offset_sec, duration_sec=usable_duration_sec, overwrite=args.overwrite)
    _mpd_df_save_fatigue_intervals(outputs.fatigue_npy, intervals, overwrite=args.overwrite)
    if args.write_validation:
        validation = _mpd_df_build_validation_json(dataset=args.dataset, subject=files.subject, files=files, outputs=outputs, eeg_info=eeg_info, psg_info=psg_info, annotation_start=annotation_start, align_start=align_start, align_end=align_end, usable_duration_sec=usable_duration_sec, intervals=intervals)
        _mpd_df_save_validation_json(outputs.validation_json, validation, overwrite=args.overwrite)
    _mpd_df_assert_outputs(outputs, include_validation=args.write_validation)
    return outputs

def _mpd_df_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess MPD-DF raw EDF files for NeuroBench.')
    parser.add_argument('--source-root', type=Path, default=_mpd_df_DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=_mpd_df_DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--validation-root', type=Path, default=_mpd_df_DEFAULT_VALIDATION_ROOT)
    parser.add_argument('--preprocess-root', type=Path, default=_mpd_df_DEFAULT_PREPROCESS_ROOT)
    parser.add_argument('--dataset', default=_mpd_df_DEFAULT_DATASET)
    parser.add_argument('--subjects', nargs='*', help='Optional subject ids, for example 1 2 42.')
    parser.add_argument('--write-validation', action='store_true', help='Write validation JSON files.')
    parser.add_argument('--recommended-subjects', action='store_true', help='Process the current recommended subset: 01, 02, 03, 04, and 06.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    args = parser.parse_args()
    if args.recommended_subjects:
        if args.subjects:
            raise SystemExit('--recommended-subjects cannot be combined with --subjects')
        args.subjects = list(_mpd_df_RECOMMENDED_SUBJECTS)
    return args

def _mpd_df_main() -> None:
    args = _mpd_df_parse_args()
    all_subjects = _mpd_df_list_subjects(args.source_root)
    selected = _mpd_df_select_subjects(all_subjects, args.subjects)
    if not selected:
        raise SystemExit(f'No MPD-DF subjects found under {args.source_root}')
    print(f'Source root: {args.source_root}')
    print(f'Output root: {args.output_root}')
    print(f'Validation root: {args.validation_root}')
    print(f'Preprocess script root: {args.preprocess_root}')
    for files in selected:
        _mpd_df_process_subject(args, files)


# FACED trial order and labels used by FacedPreprocessor.
FACED_TRIAL_METADATA = {'dataset': 'FACED',
 'description': 'Trial order and labels used to rebuild NeuroBench FACED files from official FACED '
                'pkl arrays. The pkl files contain EEG only, in video-id order.',
 'sfreq': 250.0,
 'samples_per_trial': 7500,
 'subjects': {'01': {'pkl_subject': 'sub000',
                     'output_id': 1,
                     'trials': [{'trial_id': 1,
                                 'video_id': 11,
                                 'valence': 0.8656290769577026,
                                 'arousal': 4.433903217315674,
                                 'emotion': -1},
                                {'trial_id': 2,
                                 'video_id': 8,
                                 'valence': 0.18932698667049408,
                                 'arousal': 6.748750686645508,
                                 'emotion': -1},
                                {'trial_id': 3,
                                 'video_id': 2,
                                 'valence': 0.11467285454273224,
                                 'arousal': 6.6362996101379395,
                                 'emotion': -1},
                                {'trial_id': 4,
                                 'video_id': 5,
                                 'valence': 2.532486915588379,
                                 'arousal': 2.7633137702941895,
                                 'emotion': -1},
                                {'trial_id': 5,
                                 'video_id': 6,
                                 'valence': 0.25666096806526184,
                                 'arousal': 6.254598140716553,
                                 'emotion': -1},
                                {'trial_id': 6,
                                 'video_id': 9,
                                 'valence': 0.26614582538604736,
                                 'arousal': 3.880875587463379,
                                 'emotion': -1},
                                {'trial_id': 7,
                                 'video_id': 3,
                                 'valence': 0.17434488236904144,
                                 'arousal': 4.435213088989258,
                                 'emotion': -1},
                                {'trial_id': 8,
                                 'video_id': 12,
                                 'valence': 3.155041456222534,
                                 'arousal': 2.673933982849121,
                                 'emotion': -1},
                                {'trial_id': 9,
                                 'video_id': 20,
                                 'valence': 4.771541118621826,
                                 'arousal': 3.631591796875,
                                 'emotion': 1},
                                {'trial_id': 10,
                                 'video_id': 17,
                                 'valence': 3.481172800064087,
                                 'arousal': 4.1247477531433105,
                                 'emotion': 1},
                                {'trial_id': 11,
                                 'video_id': 26,
                                 'valence': 3.5232136249542236,
                                 'arousal': 3.8023478984832764,
                                 'emotion': 1},
                                {'trial_id': 12,
                                 'video_id': 23,
                                 'valence': 4.0194172859191895,
                                 'arousal': 3.245959520339966,
                                 'emotion': 1},
                                {'trial_id': 13,
                                 'video_id': 13,
                                 'valence': 3.5539183616638184,
                                 'arousal': 0.01344401016831398,
                                 'emotion': 0},
                                {'trial_id': 14,
                                 'video_id': 15,
                                 'valence': 3.601912498474121,
                                 'arousal': 0.00905761681497097,
                                 'emotion': 0},
                                {'trial_id': 15,
                                 'video_id': 14,
                                 'valence': 3.5652546882629395,
                                 'arousal': 0.22812093794345856,
                                 'emotion': 0},
                                {'trial_id': 16,
                                 'video_id': 16,
                                 'valence': 3.5422120094299316,
                                 'arousal': 0.19744466245174408,
                                 'emotion': 0},
                                {'trial_id': 17,
                                 'video_id': 10,
                                 'valence': 1.1576944589614868,
                                 'arousal': 3.158431053161621,
                                 'emotion': -1},
                                {'trial_id': 18,
                                 'video_id': 1,
                                 'valence': 0.10638427734375,
                                 'arousal': 3.9436237812042236,
                                 'emotion': -1},
                                {'trial_id': 19,
                                 'video_id': 4,
                                 'valence': 0.03469238430261612,
                                 'arousal': 6.357564449310303,
                                 'emotion': -1},
                                {'trial_id': 20,
                                 'video_id': 7,
                                 'valence': 0.09960530698299408,
                                 'arousal': 3.268831491470337,
                                 'emotion': -1},
                                {'trial_id': 21,
                                 'video_id': 28,
                                 'valence': 5.3620524406433105,
                                 'arousal': 3.88409423828125,
                                 'emotion': 1},
                                {'trial_id': 22,
                                 'video_id': 25,
                                 'valence': 6.745361328125,
                                 'arousal': 6.643278121948242,
                                 'emotion': 1},
                                {'trial_id': 23,
                                 'video_id': 22,
                                 'valence': 4.175276756286621,
                                 'arousal': 2.826204538345337,
                                 'emotion': 1},
                                {'trial_id': 24,
                                 'video_id': 19,
                                 'valence': 3.536287546157837,
                                 'arousal': 4.3143310546875,
                                 'emotion': 1},
                                {'trial_id': 25,
                                 'video_id': 18,
                                 'valence': 4.236515522003174,
                                 'arousal': 6.086690425872803,
                                 'emotion': 1},
                                {'trial_id': 26,
                                 'video_id': 21,
                                 'valence': 6.504052639007568,
                                 'arousal': 4.666752338409424,
                                 'emotion': 1},
                                {'trial_id': 27,
                                 'video_id': 27,
                                 'valence': 5.764318943023682,
                                 'arousal': 4.215124607086182,
                                 'emotion': 1},
                                {'trial_id': 28,
                                 'video_id': 24,
                                 'valence': 3.4896321296691895,
                                 'arousal': 2.000337839126587,
                                 'emotion': 1}]},
              '02': {'pkl_subject': 'sub001',
                     'output_id': 2,
                     'trials': [{'trial_id': 1,
                                 'video_id': 20,
                                 'valence': 5.673628807067871,
                                 'arousal': 1.5316202640533447,
                                 'emotion': 1},
                                {'trial_id': 2,
                                 'video_id': 17,
                                 'valence': 5.3513712882995605,
                                 'arousal': 1.9659017324447632,
                                 'emotion': 1},
                                {'trial_id': 3,
                                 'video_id': 23,
                                 'valence': 5.977828025817871,
                                 'arousal': 4.312479496002197,
                                 'emotion': 1},
                                {'trial_id': 4,
                                 'video_id': 26,
                                 'valence': 6.143314838409424,
                                 'arousal': 1.206457495689392,
                                 'emotion': 1},
                                {'trial_id': 5,
                                 'video_id': 19,
                                 'valence': 6.126708984375,
                                 'arousal': 2.982576608657837,
                                 'emotion': 1},
                                {'trial_id': 6,
                                 'video_id': 25,
                                 'valence': 6.333268165588379,
                                 'arousal': 5.63153076171875,
                                 'emotion': 1},
                                {'trial_id': 7,
                                 'video_id': 22,
                                 'valence': 6.540625095367432,
                                 'arousal': 5.089326858520508,
                                 'emotion': 1},
                                {'trial_id': 8,
                                 'video_id': 28,
                                 'valence': 6.328539848327637,
                                 'arousal': 4.957763671875,
                                 'emotion': 1},
                                {'trial_id': 9,
                                 'video_id': 4,
                                 'valence': 1.0151937007904053,
                                 'arousal': 5.621675491333008,
                                 'emotion': -1},
                                {'trial_id': 10,
                                 'video_id': 10,
                                 'valence': 0.33684080839157104,
                                 'arousal': 5.087988376617432,
                                 'emotion': -1},
                                {'trial_id': 11,
                                 'video_id': 7,
                                 'valence': 0.658215343952179,
                                 'arousal': 5.310839653015137,
                                 'emotion': -1},
                                {'trial_id': 12,
                                 'video_id': 1,
                                 'valence': 0.6287068724632263,
                                 'arousal': 5.416825294494629,
                                 'emotion': -1},
                                {'trial_id': 13,
                                 'video_id': 3,
                                 'valence': 0.4177612364292145,
                                 'arousal': 5.6837687492370605,
                                 'emotion': -1},
                                {'trial_id': 14,
                                 'video_id': 12,
                                 'valence': 0.8413614630699158,
                                 'arousal': 1.0492879152297974,
                                 'emotion': -1},
                                {'trial_id': 15,
                                 'video_id': 6,
                                 'valence': 0.9056478142738342,
                                 'arousal': 1.1701985597610474,
                                 'emotion': -1},
                                {'trial_id': 16,
                                 'video_id': 9,
                                 'valence': 0.525769054889679,
                                 'arousal': 5.87506103515625,
                                 'emotion': -1},
                                {'trial_id': 17,
                                 'video_id': 27,
                                 'valence': 6.096032619476318,
                                 'arousal': 4.798685550689697,
                                 'emotion': 1},
                                {'trial_id': 18,
                                 'video_id': 24,
                                 'valence': 6.424328804016113,
                                 'arousal': 1.6353840827941895,
                                 'emotion': 1},
                                {'trial_id': 19,
                                 'video_id': 18,
                                 'valence': 6.424328804016113,
                                 'arousal': 1.940637230873108,
                                 'emotion': 1},
                                {'trial_id': 20,
                                 'video_id': 21,
                                 'valence': 6.438627243041992,
                                 'arousal': 5.6271443367004395,
                                 'emotion': 1},
                                {'trial_id': 21,
                                 'video_id': 5,
                                 'valence': 0.5741048455238342,
                                 'arousal': 0.9095499515533447,
                                 'emotion': -1},
                                {'trial_id': 22,
                                 'video_id': 8,
                                 'valence': 0.1996663361787796,
                                 'arousal': 5.673913478851318,
                                 'emotion': -1},
                                {'trial_id': 23,
                                 'video_id': 11,
                                 'valence': 1.12322998046875,
                                 'arousal': 0.9204019904136658,
                                 'emotion': -1},
                                {'trial_id': 24,
                                 'video_id': 2,
                                 'valence': 0.602246105670929,
                                 'arousal': 5.778902053833008,
                                 'emotion': -1},
                                {'trial_id': 25,
                                 'video_id': 13,
                                 'valence': 3.6056437492370605,
                                 'arousal': 0.17693684995174408,
                                 'emotion': 0},
                                {'trial_id': 26,
                                 'video_id': 15,
                                 'valence': 5.876570701599121,
                                 'arousal': 0.9193766117095947,
                                 'emotion': 0},
                                {'trial_id': 27,
                                 'video_id': 14,
                                 'valence': 6.643961429595947,
                                 'arousal': 0.11367594450712204,
                                 'emotion': 0},
                                {'trial_id': 28,
                                 'video_id': 16,
                                 'valence': 3.3761839866638184,
                                 'arousal': 0.21424967050552368,
                                 'emotion': 0}]},
              '03': {'pkl_subject': 'sub002',
                     'output_id': 3,
                     'trials': [{'trial_id': 1,
                                 'video_id': 28,
                                 'valence': 4.603320121765137,
                                 'arousal': 2.0277953147888184,
                                 'emotion': 1},
                                {'trial_id': 2,
                                 'video_id': 22,
                                 'valence': 0.6041259765625,
                                 'arousal': 5.035750389099121,
                                 'emotion': 1},
                                {'trial_id': 3,
                                 'video_id': 19,
                                 'valence': 5.074601173400879,
                                 'arousal': 4.3772501945495605,
                                 'emotion': 1},
                                {'trial_id': 4,
                                 'video_id': 25,
                                 'valence': 3.5629475116729736,
                                 'arousal': 0.46356201171875,
                                 'emotion': 1},
                                {'trial_id': 5,
                                 'video_id': 10,
                                 'valence': 0.9977335333824158,
                                 'arousal': 3.8194947242736816,
                                 'emotion': -1},
                                {'trial_id': 6,
                                 'video_id': 1,
                                 'valence': 0.6853312253952026,
                                 'arousal': 5.256066799163818,
                                 'emotion': -1},
                                {'trial_id': 7,
                                 'video_id': 7,
                                 'valence': 0.09504801779985428,
                                 'arousal': 6.915433883666992,
                                 'emotion': -1},
                                {'trial_id': 8,
                                 'video_id': 4,
                                 'valence': 0.02039388008415699,
                                 'arousal': 6.966162204742432,
                                 'emotion': -1},
                                {'trial_id': 9,
                                 'video_id': 3,
                                 'valence': 0.12028401345014572,
                                 'arousal': 5.375439643859863,
                                 'emotion': -1},
                                {'trial_id': 10,
                                 'video_id': 6,
                                 'valence': 0.02227376215159893,
                                 'arousal': 7.0,
                                 'emotion': -1},
                                {'trial_id': 11,
                                 'video_id': 9,
                                 'valence': 0.22199706733226776,
                                 'arousal': 6.656494140625,
                                 'emotion': -1},
                                {'trial_id': 12,
                                 'video_id': 12,
                                 'valence': 0.41909992694854736,
                                 'arousal': 0.6604369878768921,
                                 'emotion': -1},
                                {'trial_id': 13,
                                 'video_id': 21,
                                 'valence': 6.638834476470947,
                                 'arousal': 6.569819927215576,
                                 'emotion': 1},
                                {'trial_id': 14,
                                 'video_id': 27,
                                 'valence': 4.279068946838379,
                                 'arousal': 3.5900349617004395,
                                 'emotion': 1},
                                {'trial_id': 15,
                                 'video_id': 18,
                                 'valence': 4.661511421203613,
                                 'arousal': 4.563786029815674,
                                 'emotion': 1},
                                {'trial_id': 16,
                                 'video_id': 24,
                                 'valence': 4.043371677398682,
                                 'arousal': 3.5317301750183105,
                                 'emotion': 1},
                                {'trial_id': 17,
                                 'video_id': 26,
                                 'valence': 5.113936424255371,
                                 'arousal': 3.6268351078033447,
                                 'emotion': 1},
                                {'trial_id': 18,
                                 'video_id': 20,
                                 'valence': 6.574434280395508,
                                 'arousal': 6.462495803833008,
                                 'emotion': 1},
                                {'trial_id': 19,
                                 'video_id': 23,
                                 'valence': 3.4616332054138184,
                                 'arousal': 3.560497999191284,
                                 'emotion': 1},
                                {'trial_id': 20,
                                 'video_id': 17,
                                 'valence': 3.5235555171966553,
                                 'arousal': 0.40705159306526184,
                                 'emotion': 1},
                                {'trial_id': 21,
                                 'video_id': 13,
                                 'valence': 3.604475975036621,
                                 'arousal': 0.02261555939912796,
                                 'emotion': 0},
                                {'trial_id': 22,
                                 'video_id': 15,
                                 'valence': 3.5842814445495605,
                                 'arousal': 0.009883626364171505,
                                 'emotion': 0},
                                {'trial_id': 23,
                                 'video_id': 14,
                                 'valence': 3.445540428161621,
                                 'arousal': 0.01649169996380806,
                                 'emotion': 0},
                                {'trial_id': 24,
                                 'video_id': 16,
                                 'valence': 3.4065184593200684,
                                 'arousal': 0.011905924417078495,
                                 'emotion': 0},
                                {'trial_id': 25,
                                 'video_id': 2,
                                 'valence': 0.2907267212867737,
                                 'arousal': 5.962389945983887,
                                 'emotion': -1},
                                {'trial_id': 26,
                                 'video_id': 5,
                                 'valence': 0.17200927436351776,
                                 'arousal': 5.303548336029053,
                                 'emotion': -1},
                                {'trial_id': 27,
                                 'video_id': 11,
                                 'valence': 0.8766520023345947,
                                 'arousal': 4.18359375,
                                 'emotion': -1},
                                {'trial_id': 28,
                                 'video_id': 8,
                                 'valence': 0.15013428032398224,
                                 'arousal': 6.6590576171875,
                                 'emotion': -1}]},
              '04': {'pkl_subject': 'sub003',
                     'output_id': 4,
                     'trials': [{'trial_id': 1,
                                 'video_id': 2,
                                 'valence': 1.515100121498108,
                                 'arousal': 4.595971584320068,
                                 'emotion': -1},
                                {'trial_id': 2,
                                 'video_id': 8,
                                 'valence': 1.4174600839614868,
                                 'arousal': 5.689009666442871,
                                 'emotion': -1},
                                {'trial_id': 3,
                                 'video_id': 11,
                                 'valence': 0.40044352412223816,
                                 'arousal': 1.0640422105789185,
                                 'emotion': -1},
                                {'trial_id': 4,
                                 'video_id': 5,
                                 'valence': 0.46609699726104736,
                                 'arousal': 4.620267868041992,
                                 'emotion': -1},
                                {'trial_id': 5,
                                 'video_id': 25,
                                 'valence': 6.666178226470947,
                                 'arousal': 4.7771525382995605,
                                 'emotion': 1},
                                {'trial_id': 6,
                                 'video_id': 28,
                                 'valence': 4.3552327156066895,
                                 'arousal': 0.6304158568382263,
                                 'emotion': 1},
                                {'trial_id': 7,
                                 'video_id': 19,
                                 'valence': 4.21478271484375,
                                 'arousal': 3.4821696281433105,
                                 'emotion': 1},
                                {'trial_id': 8,
                                 'video_id': 22,
                                 'valence': 4.4793620109558105,
                                 'arousal': 1.9514892101287842,
                                 'emotion': 1},
                                {'trial_id': 9,
                                 'video_id': 24,
                                 'valence': 6.822009086608887,
                                 'arousal': 4.649434566497803,
                                 'emotion': 1},
                                {'trial_id': 10,
                                 'video_id': 21,
                                 'valence': 5.500764846801758,
                                 'arousal': 4.195813179016113,
                                 'emotion': 1},
                                {'trial_id': 11,
                                 'video_id': 27,
                                 'valence': 3.8737549781799316,
                                 'arousal': 1.9367350339889526,
                                 'emotion': 1},
                                {'trial_id': 12,
                                 'video_id': 18,
                                 'valence': 5.4714274406433105,
                                 'arousal': 4.76934814453125,
                                 'emotion': 1},
                                {'trial_id': 13,
                                 'video_id': 4,
                                 'valence': 0.014982095919549465,
                                 'arousal': 6.956335544586182,
                                 'emotion': -1},
                                {'trial_id': 14,
                                 'video_id': 7,
                                 'valence': 1.2485555410385132,
                                 'arousal': 3.8354451656341553,
                                 'emotion': -1},
                                {'trial_id': 15,
                                 'video_id': 10,
                                 'valence': 1.7483195066452026,
                                 'arousal': 4.629781246185303,
                                 'emotion': -1},
                                {'trial_id': 16,
                                 'video_id': 1,
                                 'valence': 2.329345703125,
                                 'arousal': 3.9066529273986816,
                                 'emotion': -1},
                                {'trial_id': 17,
                                 'video_id': 6,
                                 'valence': 0.8727498650550842,
                                 'arousal': 4.3781046867370605,
                                 'emotion': -1},
                                {'trial_id': 18,
                                 'video_id': 12,
                                 'valence': 1.1556437015533447,
                                 'arousal': 3.852250099182129,
                                 'emotion': -1},
                                {'trial_id': 19,
                                 'video_id': 3,
                                 'valence': 1.376928687095642,
                                 'arousal': 4.9211344718933105,
                                 'emotion': -1},
                                {'trial_id': 20,
                                 'video_id': 9,
                                 'valence': 1.2655599117279053,
                                 'arousal': 4.734969139099121,
                                 'emotion': -1},
                                {'trial_id': 21,
                                 'video_id': 14,
                                 'valence': 4.367765426635742,
                                 'arousal': 1.753759741783142,
                                 'emotion': 0},
                                {'trial_id': 22,
                                 'video_id': 13,
                                 'valence': 2.326953172683716,
                                 'arousal': 1.4217040538787842,
                                 'emotion': 0},
                                {'trial_id': 23,
                                 'video_id': 16,
                                 'valence': 3.011914014816284,
                                 'arousal': 1.6701619625091553,
                                 'emotion': 0},
                                {'trial_id': 24,
                                 'video_id': 15,
                                 'valence': 3.083833932876587,
                                 'arousal': 1.67352294921875,
                                 'emotion': 0},
                                {'trial_id': 25,
                                 'video_id': 26,
                                 'valence': 3.8837523460388184,
                                 'arousal': 3.858658790588379,
                                 'emotion': 1},
                                {'trial_id': 26,
                                 'video_id': 23,
                                 'valence': 6.282910346984863,
                                 'arousal': 4.109822750091553,
                                 'emotion': 1},
                                {'trial_id': 27,
                                 'video_id': 17,
                                 'valence': 3.607666015625,
                                 'arousal': 3.963789939880371,
                                 'emotion': 1},
                                {'trial_id': 28,
                                 'video_id': 20,
                                 'valence': 5.393754005432129,
                                 'arousal': 3.7930338382720947,
                                 'emotion': 1}]},
              '05': {'pkl_subject': 'sub004',
                     'output_id': 5,
                     'trials': [{'trial_id': 1,
                                 'video_id': 25,
                                 'valence': 6.124715328216553,
                                 'arousal': 3.502734422683716,
                                 'emotion': 1},
                                {'trial_id': 2,
                                 'video_id': 19,
                                 'valence': 3.357499122619629,
                                 'arousal': 1.2108724117279053,
                                 'emotion': 1},
                                {'trial_id': 3,
                                 'video_id': 22,
                                 'valence': 3.626664161682129,
                                 'arousal': 4.294507026672363,
                                 'emotion': 1},
                                {'trial_id': 4,
                                 'video_id': 28,
                                 'valence': 5.226330757141113,
                                 'arousal': 2.779891014099121,
                                 'emotion': 1},
                                {'trial_id': 5,
                                 'video_id': 27,
                                 'valence': 5.947094917297363,
                                 'arousal': 4.4413371086120605,
                                 'emotion': 1},
                                {'trial_id': 6,
                                 'video_id': 21,
                                 'valence': 5.868595600128174,
                                 'arousal': 3.8055949211120605,
                                 'emotion': 1},
                                {'trial_id': 7,
                                 'video_id': 18,
                                 'valence': 3.5678181648254395,
                                 'arousal': 4.117455959320068,
                                 'emotion': 1},
                                {'trial_id': 8,
                                 'video_id': 24,
                                 'valence': 4.263118267059326,
                                 'arousal': 1.1291544437408447,
                                 'emotion': 1},
                                {'trial_id': 9,
                                 'video_id': 5,
                                 'valence': 0.03682861477136612,
                                 'arousal': 5.186852931976318,
                                 'emotion': -1},
                                {'trial_id': 10,
                                 'video_id': 11,
                                 'valence': 0.0,
                                 'arousal': 4.798856735229492,
                                 'emotion': -1},
                                {'trial_id': 11,
                                 'video_id': 8,
                                 'valence': 0.04093017429113388,
                                 'arousal': 6.884273052215576,
                                 'emotion': -1},
                                {'trial_id': 12,
                                 'video_id': 2,
                                 'valence': 0.02398274652659893,
                                 'arousal': 5.775996685028076,
                                 'emotion': -1},
                                {'trial_id': 13,
                                 'video_id': 7,
                                 'valence': 0.06160888820886612,
                                 'arousal': 6.4865641593933105,
                                 'emotion': -1},
                                {'trial_id': 14,
                                 'video_id': 1,
                                 'valence': 0.013102213852107525,
                                 'arousal': 4.198319435119629,
                                 'emotion': -1},
                                {'trial_id': 15,
                                 'video_id': 4,
                                 'valence': 1.5586222410202026,
                                 'arousal': 5.988679885864258,
                                 'emotion': -1},
                                {'trial_id': 16,
                                 'video_id': 10,
                                 'valence': 2.509187936782837,
                                 'arousal': 2.481900930404663,
                                 'emotion': -1},
                                {'trial_id': 17,
                                 'video_id': 6,
                                 'valence': 0.02663167379796505,
                                 'arousal': 6.9578166007995605,
                                 'emotion': -1},
                                {'trial_id': 18,
                                 'video_id': 3,
                                 'valence': 0.02497965469956398,
                                 'arousal': 6.186751365661621,
                                 'emotion': -1},
                                {'trial_id': 19,
                                 'video_id': 12,
                                 'valence': 1.5439534187316895,
                                 'arousal': 2.702559471130371,
                                 'emotion': -1},
                                {'trial_id': 20,
                                 'video_id': 9,
                                 'valence': 0.04702555388212204,
                                 'arousal': 6.990230083465576,
                                 'emotion': -1},
                                {'trial_id': 21,
                                 'video_id': 13,
                                 'valence': 3.550870656967163,
                                 'arousal': 3.5817179679870605,
                                 'emotion': 0},
                                {'trial_id': 22,
                                 'video_id': 15,
                                 'valence': 3.40045166015625,
                                 'arousal': 1.6989867687225342,
                                 'emotion': 0},
                                {'trial_id': 23,
                                 'video_id': 14,
                                 'valence': 3.45556640625,
                                 'arousal': 0.3095540404319763,
                                 'emotion': 0},
                                {'trial_id': 24,
                                 'video_id': 16,
                                 'valence': 3.5932536125183105,
                                 'arousal': 0.5564737915992737,
                                 'emotion': 0},
                                {'trial_id': 25,
                                 'video_id': 20,
                                 'valence': 6.576428413391113,
                                 'arousal': 4.975736618041992,
                                 'emotion': 1},
                                {'trial_id': 26,
                                 'video_id': 23,
                                 'valence': 6.958357810974121,
                                 'arousal': 6.210847854614258,
                                 'emotion': 1},
                                {'trial_id': 27,
                                 'video_id': 17,
                                 'valence': 3.0668578147888184,
                                 'arousal': 4.231245994567871,
                                 'emotion': 1},
                                {'trial_id': 28,
                                 'video_id': 26,
                                 'valence': 6.963313579559326,
                                 'arousal': 6.7426838874816895,
                                 'emotion': 1}]}}}


def build_runs(args: argparse.Namespace, metadata_file: Path) -> list[tuple[str, type, list[str]]]:
    runs: list[tuple[str, type, list[str]]] = []
    if args.faced_pkl_root and not args.skip_faced:
        argv = ["--pkl-root", str(args.faced_pkl_root), "--metadata-file", str(metadata_file), "--output-root", str(args.output_root), "--no-progress"]
        add_subject_args(argv, "--subjects", args.faced_subjects)
        if args.overwrite:
            argv.append("--overwrite")
        runs.append(("FACED", FacedPreprocessor, argv))
    if args.refed_root and not args.skip_refed:
        argv = ["--source-root", str(args.refed_root), "--output-root", str(args.output_root)]
        add_subject_args(argv, "--subjects", args.refed_subjects)
        if args.overwrite:
            argv.append("--overwrite")
        runs.append(("REFED", RefedPreprocessor, argv))
    if args.cog_bci_root and not args.skip_cog_bci:
        argv = ["--source-root", str(args.cog_bci_root), "--output-root", str(args.output_root)]
        add_subject_args(argv, "--subjects", args.cog_bci_subjects)
        if args.overwrite:
            argv.append("--overwrite")
        runs.append(("COG-BCI", CogBciPreprocessor, argv))
    if args.mpd_df_root and not args.skip_mpd_df:
        argv = ["--source-root", str(args.mpd_df_root), "--output-root", str(args.output_root), "--preprocess-root", str(Path(__file__).resolve().parent)]
        add_subject_args(argv, "--subjects", args.mpd_df_subjects)
        if args.overwrite:
            argv.append("--overwrite")
        runs.append(("MPD-DF", MpdDfPreprocessor, argv))
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing FACED, REFED, COG-BCI, and MPD-DF subdirectories.",
    )
    parser.add_argument("--faced-pkl-root", type=Path, default=None)
    parser.add_argument("--refed-root", type=Path, default=None)
    parser.add_argument("--cog-bci-root", type=Path, default=None)
    parser.add_argument("--mpd-df-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--faced-subjects", nargs="*")
    parser.add_argument("--refed-subjects", nargs="*")
    parser.add_argument("--cog-bci-subjects", nargs="*")
    parser.add_argument("--mpd-df-subjects", nargs="*")
    parser.add_argument("--skip-faced", action="store_true")
    parser.add_argument("--skip-refed", action="store_true")
    parser.add_argument("--skip-cog-bci", action="store_true")
    parser.add_argument("--skip-mpd-df", action="store_true")
    args = parser.parse_args(argv)
    args.output_root = DEFAULT_OUTPUT_ROOT
    if args.faced_pkl_root is None:
        args.faced_pkl_root = args.data_root / "FACED"
    if args.refed_root is None:
        args.refed_root = args.data_root / "REFED"
    if args.cog_bci_root is None:
        args.cog_bci_root = args.data_root / "COG-BCI"
    if args.mpd_df_root is None:
        args.mpd_df_root = args.data_root / "MPD-DF"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="neurobench_faced_metadata_") as temp_dir:
        metadata_file = Path(temp_dir) / "faced_trial_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as file_obj:
            json.dump(FACED_TRIAL_METADATA, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        runs = build_runs(args, metadata_file)
        if not runs:
            raise SystemExit("No preprocessing commands selected.")
        for label, pipeline_cls, argv in progress_iter(runs):
            print(f"\n[{label}] RUN {pipeline_cls.__name__} {' '.join(argv)}", flush=True)
            if args.dry_run:
                continue
            pipeline_cls.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

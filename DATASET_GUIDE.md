# BrainBench Dataset Access Guide

This document lists the raw EEG/PSG datasets required by the currently released BrainBench subsets, their official access pages, and the records selected by the BrainBench preparation scripts.

BrainBench does not redistribute raw EEG/PSG recordings. Users must obtain the data from the original providers and comply with each dataset's access policy, license, and data-use requirements.

## Benchmark case files

The fixed evaluation case JSON files are published in the [BrainBench Hugging Face dataset](https://huggingface.co/datasets/xbb083/BrainBench). Raw EEG/PSG files are not included there.

## Local data layout

After downloading the source data, organize it as follows:

```text
downloads/raw/
├── foundational_analysis/
│   └── original/
│       ├── isruc/
│       ├── bcic2020-3/
│       ├── MentalArithmetic/
│       ├── mumtaz/
│       └── seedv/
└── sleep_assessment/
    └── original/
        ├── isruc/
        ├── hmc/
        ├── shhs/
        ├── mass/
        └── physionet2018/
```

The preparation scripts also accept the same folders without the intermediate `original/` directory.

## Foundational Analysis

The current preparation script uses five records from each source dataset and writes standardized files to `data/core/`.

| Dataset | Official access | Records used by BrainBench | Local folder |
|---|---|---|---|
| ISRUC-Sleep | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/) | Subjects 1–5 (`1/1.rec` … `5/5.rec`) | `isruc/` |
| BCI Competition 2020 Track 3 | [Competition page](https://brain.korea.ac.kr/bci2020/competition.php) · [OSF repository](https://osf.io/pq7vb/) | `Data_Sample01.mat` … `Data_Sample05.mat` | `bcic2020-3/` |
| EEG During Mental Arithmetic Tasks | [PhysioNet EEGMAT](https://physionet.org/content/eegmat/1.0.0/) | `Subject01_1.edf` … `Subject05_1.edf` | `MentalArithmetic/` |
| Mumtaz 2016 EEG | [Figshare EEG Data New](https://figshare.com/articles/dataset/EEG_Data_New/4244171) | `H S1 EC.edf` … `H S5 EC.edf` | `mumtaz/` |
| SEED-V | [SJTU BCMI SEED portal](https://bcmi.sjtu.edu.cn/home/seed/) | Recordings `1_1_20180804.cnt`, `2_1_20180416.cnt`, `3_1_20180414.cnt`, `4_1_20180414.cnt`, `5_1_20180719.cnt` | `seedv/` |

## Sleep Assessment

The current preparation script uses five records from each source dataset and writes standardized files to `data/sleep/`.

| Dataset | Official access | Records used by BrainBench | Local folder |
|---|---|---|---|
| ISRUC-Sleep | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/) | Subjects 1–5 (`1/1.rec` … `5/5.rec`) | `isruc/` |
| HMC Sleep Staging Database | [PhysioNet HMC](https://physionet.org/content/hmc-sleep-staging/1.1/) | `SN001` … `SN005` | `hmc/` |
| Sleep Heart Health Study (SHHS1) | [NSRR SHHS](https://sleepdata.org/datasets/shhs) | `shhs1-200001` … `shhs1-200005` | `shhs/` |
| Montreal Archive of Sleep Studies (MASS) | [MASS access page](http://www.ceams-carsm.ca/en/MASS) | `01-03-0001` … `01-03-0005` | `mass/` |
| PhysioNet/CinC Challenge 2018 | [PhysioNet Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/) | `tr03-0005`, `tr03-0029`, `tr03-0052`, `tr03-0061`, `tr03-0078` | `physionet2018/` |

## Important notes

- The record lists above are the exact inputs currently selected by `prepare_core_test_inputs.py` and `prepare_sleep_test_inputs.py`; they are not the full source datasets.
- Do not rename the local folders or the source files expected by the preparation scripts.
- The `neurocognitive_assessment` and `physiological_integration` subsets are still in progress and will receive their dataset lists when their manifests and preparation scripts are released.
- Keep raw downloads under `downloads/`; do not commit them to the public GitHub repository.

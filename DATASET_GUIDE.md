# BrainBench Dataset Access Guide

This document lists the raw EEG/PSG datasets required by BrainBench, their official access pages, and the records selected by the preparation scripts.

## Benchmark case files

The fixed evaluation case JSON files are published in the [BrainBench Hugging Face dataset](https://huggingface.co/datasets/xbb083/BrainBench). 

## Local data layout

Choose one `<data-root>` for each subset and place its required dataset folders directly underneath it. The root directory may be located anywhere on the user's system.

### Foundational Analysis

```text
<foundational-data-root>/
├── isruc/
│   ├── 1/
│   │   ├── 1.rec
│   │   ├── 1_1.txt
│   │   └── 1_1.xlsx
│   ├── 2/
│   │   ├── 2.rec
│   │   ├── 2_1.txt
│   │   └── 2_1.xlsx
│   ├── 3/
│   │   ├── 3.rec
│   │   ├── 3_1.txt
│   │   └── 3_1.xlsx
│   ├── 4/
│   │   ├── 4.rec
│   │   ├── 4_1.txt
│   │   └── 4_1.xlsx
│   └── 5/
│       ├── 5.rec
│       ├── 5_1.txt
│       └── 5_1.xlsx
│
├── bcic2020-3/
│   └── Training set/
│       ├── Data_Sample01.mat
│       ├── Data_Sample02.mat
│       ├── Data_Sample03.mat
│       ├── Data_Sample04.mat
│       └── Data_Sample05.mat
│
├── MentalArithmetic/
│   └── edf/
│       ├── Subject00_1.edf
│       ├── Subject01_1.edf
│       ├── Subject02_1.edf
│       ├── Subject03_1.edf
│       └── Subject04_1.edf
│
├── mumtaz/
│   └── files/
│       ├── H S1 EC.edf
│       ├── H S2 EC.edf
│       ├── H S3 EC.edf
│       ├── H S4 EC.edf
│       └── H S5 EC.edf
│
└── seedv/
    └── files/
        ├── 1_1_20180804.cnt
        ├── 2_1_20180416.cnt
        ├── 3_1_20180414.cnt
        ├── 4_1_20180414.cnt
        └── 5_1_20180719.cnt
```
For Foundational Analysis, the preparation script reads the ISRUC `.rec` files. The corresponding `.txt` and `.xlsx` files may remain in the subject directories but are not used by this subset. Labels required by BCI Competition 2020 Track 3 are stored inside the corresponding `.mat` files.

Prepare the subset with:

```bash
python main.py prepare foundational_analysis \
  --data-root /path/to/foundational-data-root
```

### Sleep Assessment

```text
<sleep-data-root>/
├── isruc/
│   ├── 1/
│   │   ├── 1.rec
│   │   └── 1_1.txt
│   ├── 2/
│   │   ├── 2.rec
│   │   └── 2_1.txt
│   ├── 3/
│   │   ├── 3.rec
│   │   └── 3_1.txt
│   ├── 4/
│   │   ├── 4.rec
│   │   └── 4_1.txt
│   └── 5/
│       ├── 5.rec
│       └── 5_1.txt
│
├── hmc/
│   ├── SN001.edf
│   ├── SN001_sleepscoring.edf
│   ├── SN002.edf
│   ├── SN002_sleepscoring.edf
│   ├── SN003.edf
│   ├── SN003_sleepscoring.edf
│   ├── SN004.edf
│   ├── SN004_sleepscoring.edf
│   ├── SN005.edf
│   └── SN005_sleepscoring.edf
│
├── mass/
│   ├── 01-03-0001 PSG.edf
│   ├── 01-03-0001 Base.edf
│   ├── 01-03-0002 PSG.edf
│   ├── 01-03-0002 Base.edf
│   ├── 01-03-0003 PSG.edf
│   ├── 01-03-0003 Base.edf
│   ├── 01-03-0004 PSG.edf
│   ├── 01-03-0004 Base.edf
│   ├── 01-03-0005 PSG.edf
│   └── 01-03-0005 Base.edf
│
├── physionet2018/
│   ├── tr03-0005/
│   │   ├── tr03-0005.mat
│   │   ├── tr03-0005.hea
│   │   └── tr03-0005.arousal
│   ├── tr03-0029/
│   │   ├── tr03-0029.mat
│   │   ├── tr03-0029.hea
│   │   └── tr03-0029.arousal
│   ├── tr03-0052/
│   │   ├── tr03-0052.mat
│   │   ├── tr03-0052.hea
│   │   └── tr03-0052.arousal
│   ├── tr03-0061/
│   │   ├── tr03-0061.mat
│   │   ├── tr03-0061.hea
│   │   └── tr03-0061.arousal
│   └── tr03-0078/
│       ├── tr03-0078.mat
│       ├── tr03-0078.hea
│       └── tr03-0078.arousal
│
└── shhs/
    ├── shhs1-200001.edf
    ├── shhs1-200001-profusion.xml
    ├── shhs1-200002.edf
    ├── shhs1-200002-profusion.xml
    ├── shhs1-200003.edf
    ├── shhs1-200003-profusion.xml
    ├── shhs1-200004.edf
    ├── shhs1-200004-profusion.xml
    ├── shhs1-200005.edf
    └── shhs1-200005-profusion.xml
```

Prepare the subset with:

```bash
python main.py prepare sleep_assessment \
  --data-root /path/to/sleep-data-root
```


## Foundational Analysis

The current preparation script uses five records from each source dataset and writes standardized files to `data/core/`.

| Dataset | Official access | Records used by BrainBench | Local folder |
|---|---|---|---|
| ISRUC-Sleep | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/) | Subjects 1–5 (`1/1.rec` … `5/5.rec`) | `isruc/` |
| BCI Competition 2020 Track 3 | [Competition page](https://brain.korea.ac.kr/bci2020/competition.php) · [OSF repository](https://osf.io/pq7vb/) | `Data_Sample01.mat` … `Data_Sample05.mat` | `bcic2020-3/` |
| EEG During Mental Arithmetic Tasks | [PhysioNet EEGMAT](https://physionet.org/content/eegmat/1.0.0/) | `Subject00_1.edf` … `Subject04_1.edf` | `MentalArithmetic/` |
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

## Neurocognitive Assessment

The current preparation script uses five records from each source dataset and writes standardized files to `data/emotion/`.

```text
<emotion-data-root>/
├── FACED/
│   ├── sub000.pkl
│   ├── sub001.pkl
│   ├── sub002.pkl
│   ├── sub003.pkl
│   └── sub004.pkl
├── REFED/
│   ├── data/
│   │   ├── 1/EEG_videos.mat
│   │   ├── 2/EEG_videos.mat
│   │   ├── 3/EEG_videos.mat
│   │   ├── 4/EEG_videos.mat
│   │   └── 5/EEG_videos.mat
│   ├── annotations/
│   │   ├── 1_label.mat
│   │   ├── 2_label.mat
│   │   ├── 3_label.mat
│   │   ├── 4_label.mat
│   │   └── 5_label.mat
│   ├── EEG_channels.csv
│   └── SAM_score.csv
├── COG-BCI/
│   ├── sub-01/ses-S1/
│   │   ├── eeg/
│   │   │   ├── zeroBACK.set
│   │   │   ├── zeroBACK.fdt
│   │   │   ├── oneBACK.set
│   │   │   ├── oneBACK.fdt
│   │   │   ├── twoBACK.set
│   │   │   └── twoBACK.fdt
│   │   └── behavioral/
│   │       ├── 0-Back.mat
│   │       ├── 1-Back.mat
│   │       └── 2-Back.mat
│   ├── sub-02/ses-S1/
│   │   ├── eeg/
│   │   │   ├── zeroBACK.set
│   │   │   ├── zeroBACK.fdt
│   │   │   ├── oneBACK.set
│   │   │   ├── oneBACK.fdt
│   │   │   ├── twoBACK.set
│   │   │   └── twoBACK.fdt
│   │   └── behavioral/
│   │       ├── 0-Back.mat
│   │       ├── 1-Back.mat
│   │       └── 2-Back.mat
│   ├── sub-03/ses-S1/
│   │   ├── eeg/
│   │   │   ├── zeroBACK.set
│   │   │   ├── zeroBACK.fdt
│   │   │   ├── oneBACK.set
│   │   │   ├── oneBACK.fdt
│   │   │   ├── twoBACK.set
│   │   │   └── twoBACK.fdt
│   │   └── behavioral/
│   │       ├── 0-Back.mat
│   │       ├── 1-Back.mat
│   │       └── 2-Back.mat
│   ├── sub-04/ses-S1/
│   │   ├── eeg/
│   │   │   ├── zeroBACK.set
│   │   │   ├── zeroBACK.fdt
│   │   │   ├── oneBACK.set
│   │   │   ├── oneBACK.fdt
│   │   │   ├── twoBACK.set
│   │   │   └── twoBACK.fdt
│   │   └── behavioral/
│   │       ├── 0-Back.mat
│   │       ├── 1-Back.mat
│   │       └── 2-Back.mat
│   └── sub-05/ses-S1/
│       ├── eeg/
│       │   ├── zeroBACK.set
│       │   ├── zeroBACK.fdt
│       │   ├── oneBACK.set
│       │   ├── oneBACK.fdt
│       │   ├── twoBACK.set
│       │   └── twoBACK.fdt
│       └── behavioral/
│           ├── 0-Back.mat
│           ├── 1-Back.mat
│           └── 2-Back.mat
└── MPD-DF/
    ├── EEG/
    │   ├── MPDDF_raw_01_EEG.edf
    │   ├── MPDDF_raw_02_EEG.edf
    │   ├── MPDDF_raw_03_EEG.edf
    │   ├── MPDDF_raw_04_EEG.edf
    │   └── MPDDF_raw_06_EEG.edf
    ├── PSG/
    │   ├── MPDDF_raw_01_PSG.edf
    │   ├── MPDDF_raw_02_PSG.edf
    │   ├── MPDDF_raw_03_PSG.edf
    │   ├── MPDDF_raw_04_PSG.edf
    │   └── MPDDF_raw_06_PSG.edf
    └── Annotation/
        ├── MPDDF_raw_01_Annotation.txt
        ├── MPDDF_raw_02_Annotation.txt
        ├── MPDDF_raw_03_Annotation.txt
        ├── MPDDF_raw_04_Annotation.txt
        └── MPDDF_raw_06_Annotation.txt
```

| Dataset | Official access | Records used by BrainBench | Local folder |
|---|---|---|---|
| FACED | [FACED](https://www.synapse.org/Synapse:syn50614194/files/) | `sub000.pkl` … `sub004.pkl` | `FACED/` |
| REFED | [REFED](https://huggingface.co/datasets/REFED2025/REFED-dataset) | Subjects `1` … `5`, their annotations, `EEG_channels.csv`, and `SAM_score.csv` | `REFED/` |
| COG-BCI | [COG-BCI](https://zenodo.org/records/6874129) | Subjects `sub-01` … `sub-05`, including EEG and behavioral files for the 0-, 1-, and 2-back tasks | `COG-BCI/` |
| MPD-DF | [MPD-DF](https://figshare.com/articles/dataset/MPD-DF_Multimodal_Phenotyping_Dataset_of_Driving_Fatigue_--_The_Raw_Dataset_and_Questionnaire_Information/28455737) | Subjects `01`, `02`, `03`, `04`, and `06`, with Annotation, EEG, and PSG files | `MPD-DF/` |

## Physiological Integration

The preparation workflow uses five records from each source dataset and writes standardized files to `data/multi/`.

```text
<multi-data-root>/
├── DEAP/
│   ├── data_original/
│   │   ├── s01.bdf
│   │   ├── s02.bdf
│   │   ├── s03.bdf
│   │   ├── s04.bdf
│   │   └── s05.bdf
│   └── data_preprocessed_matlab/
│       ├── s01.mat
│       ├── s02.mat
│       ├── s03.mat
│       ├── s04.mat
│       └── s05.mat
├── SEED-VIG/
│   ├── Raw_Data/
│   │   ├── 1_20151124_noon_2.mat
│   │   ├── 2_20151106_noon.mat
│   │   ├── 3_20151024_noon.mat
│   │   ├── 4_20151105_noon.mat
│   │   └── 5_20141108_noon.mat
│   └── perclos_labels/
│       ├── 1_20151124_noon_2.mat
│       ├── 2_20151106_noon.mat
│       ├── 3_20151024_noon.mat
│       ├── 4_20151105_noon.mat
│       └── 5_20141108_noon.mat
├── SEED-VII/
│   ├── EEG_raw/
│   │   ├── 1_20221001_1.cnt
│   │   ├── 2_20221011_1.cnt
│   │   ├── 3_20221009_1.cnt
│   │   ├── 4_20221030_1.cnt
│   │   └── 5_20220929_1.cnt
│   ├── EYE_raw/
│   │   ├── 1_20221001_1.tsv
│   │   ├── 2_20221011_1.tsv
│   │   ├── 3_20221009_1.tsv
│   │   ├── 4_20221030_1.tsv
│   │   └── 5_20220929_1.tsv
│   ├── save_info/
│   │   ├── 1_20221001_1_save_info.csv
│   │   ├── 1_20221001_1_trigger_info.csv
│   │   ├── 2_20221011_1_save_info.csv
│   │   ├── 2_20221011_1_trigger_info.csv
│   │   ├── 3_20221009_1_save_info.csv
│   │   ├── 3_20221009_1_trigger_info.csv
│   │   ├── 4_20221030_1_save_info.csv
│   │   ├── 4_20221030_1_trigger_info.csv
│   │   ├── 5_20220929_1_save_info.csv
│   │   └── 5_20220929_1_trigger_info.csv
│   └── continuous_labels/
│       ├── 1.mat
│       ├── 2.mat
│       ├── 3.mat
│       ├── 4.mat
│       └── 5.mat
├── SHHS/
│   ├── shhs1-200001.edf
│   ├── shhs1-200001-profusion.xml
│   ├── shhs1-200002.edf
│   ├── shhs1-200002-profusion.xml
│   ├── shhs1-200003.edf
│   ├── shhs1-200003-profusion.xml
│   ├── shhs1-200004.edf
│   ├── shhs1-200004-profusion.xml
│   ├── shhs1-200005.edf
│   └── shhs1-200005-profusion.xml
└── simultaneous_EEG-NIRS/
    ├── EEG/
    │   ├── VP001.zip
    │   ├── VP002.zip
    │   ├── VP003.zip
    │   ├── VP004.zip
    │   └── VP005.zip
    └── NIRS/
        ├── VP001.zip
        ├── VP002.zip
        ├── VP003.zip
        ├── VP004.zip
        └── VP005.zip
```

| Dataset | Official access | Records used by BrainBench | Local folder |
|---|---|---|---|
| DEAP | [Official DEAP page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) · [Kaggle BDF](https://www.kaggle.com/datasets/sayuksh/deap-datasetraw-data) · [Kaggle MATLAB](https://www.kaggle.com/datasets/lilnoah03/deapmatlab) | Subjects `s01` … `s05`, with paired BDF and MATLAB files | `DEAP/` |
| SEED-VIG | [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html) | Five listed raw recordings and matching PERCLOS labels | `SEED-VIG/` |
| SEED-VII | [SEED-VII](https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html) | Five EEG and eye-tracking sessions, save/trigger metadata, and continuous labels | `SEED-VII/` |
| SHHS1 | [NSRR SHHS](https://sleepdata.org/datasets/shhs) | `shhs1-200001` … `shhs1-200005`, each with EDF and Profusion XML | `SHHS/` |
| Simultaneous EEG–NIRS | [TU Berlin download page](https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/) | Participants `VP001` … `VP005`, with paired EEG and NIRS archives | `simultaneous_EEG-NIRS/` |

## Important notes

- Do not rename the local folders or the source files expected by the preparation scripts.
- Keep raw datasets outside Git tracking and do not commit them to the public GitHub repository.

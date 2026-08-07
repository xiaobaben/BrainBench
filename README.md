<h2 align="center">
  <img src="assets/brainbench_wordmark.svg" alt="BrainBench" width="520">
  <br>
  Benchmarking LLMs for Comprehensive EEG Understanding
</h2>
<br>
<p align="center">
  <img src="assets/intro.png" alt="BrainBench overview" width="100%">
</p>

<p align="center">
  <b>BrainBench</b> is a benchmark for comprehensive EEG understanding, evaluating how large language models and agentic systems analyze real-world EEG recordings and produce scientifically grounded conclusions.
</p>

<p align="center">
  <a href="#">&#128196; ArXiv Paper</a>
  &nbsp;|&nbsp;
  <a href="#">&#127760; Website</a>
  &nbsp;|&nbsp;
  <a href="#">&#129303; Hugging Face</a>
  &nbsp;|&nbsp;
  <a href="#quickstart">&#128640; Quickstart</a>
  &nbsp;|&nbsp;
  <a href="#">&#128218; Citation</a>
</p>
<hr>

<h2>&#128300; Overview</h2>

<p>
  BrainBench evaluates EEG understanding from multiple complementary perspectives through four benchmark subsets:
</p>

- **Foundational Analysis:** Core EEG signal understanding and analysis.
- **Sleep Assessment:** Sleep-related EEG assessment and staging.
- **Neurocognitive Assessment:** EEG-based assessment of cognitive functions.
- **Physiological Integration:** Joint reasoning over EEG and physiological information.

<h3>&#128207; Benchmark at a glance</h3>

<table>
  <thead>
    <tr>
      <th>Item</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Subsets</td>
      <td>4</td>
    </tr>
    <tr>
      <td>Datasets</td>
      <td>17</td>
    </tr>
    <tr>
      <td>Tasks</td>
      <td>172</td>
    </tr>
    <tr>
      <td>Evaluation instances</td>
      <td>~4,000</td>
    </tr>
  </tbody>
</table>
<hr>

<h2>&#128193; Project structure</h2>

```text
BrainBench/
├── brainbench/                  # core benchmark package
│   ├── codeact/                 # CodeAct agent execution modules
│   ├── agent.py                 # agent interface
│   ├── cases.py                 # benchmark case loading
│   ├── config.py                # runtime configuration
│   ├── evaluator.py             # evaluation pipeline
│   ├── llm.py                   # LLM request layer
│   ├── runners.py               # evaluation runners
│   └── scoring.py               # scoring and aggregation
├── benchmarks/                  # evaluation cases downloaded from Hugging Face
│   ├── foundational_analysis/
│   │   └── cases/               # evaluation JSON files
│   ├── sleep_assessment/
│   │   └── cases/               # evaluation JSON files
│   ├── neurocognitive_assessment/
│   │   └── cases/               # Neurocognitive Assessment cases
│   └── physiological_integration/
│       └── cases/               # Physiological Integration cases
├── docker/                      # Docker environments
│   └── codeact/                 # CodeAct Docker image definition
├── examples/                    # examples and offline smoke tests
│   └── synthetic_smoke/
├── assets/                      # README images and project artwork
├── main.py                      # command-line entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # environment variable template
└── .gitignore                   # ignored local files
```

> The `benchmarks/` directory is intentionally shipped without evaluation JSON files. Download the corresponding case files from Hugging Face and place them under `benchmarks/<subset>/cases/` before running an evaluation.
<hr>

<h2 id="quickstart">&#128640; Quickstart</h2>

<p>Run these commands from the repository root and use the identifier of the subset you want to operate on.</p>

### 1. Install

```bash
git clone https://github.com/xiaobaben/BrainBench.git
cd BrainBench
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in the model credentials and choose the CodeAct execution mode in `.env`:

```dotenv
BRAINBENCH_API_KEY=YOUR_API_KEY
BRAINBENCH_BASE_URL=https://YOUR_PROVIDER_BASE_URL/v1
BRAINBENCH_MODEL=YOUR_MODEL_NAME
```

### 3. Smoke test

```bash
python main.py smoke
```

### 4. Download benchmark cases

The fixed case JSON files are published in the [BrainBench Hugging Face dataset](https://huggingface.co/datasets/xbb083/BrainBench). They contain the benchmark inputs, parsing instructions, ground truth, and metrics; raw EEG/PSG recordings are not included.

Install the Hugging Face CLI and download all benchmark cases:

```bash
python -m pip install --upgrade huggingface_hub
hf download xbb083/BrainBench \
  --repo-type dataset \
  --local-dir ./benchmarks
```

To download only the Foundational Analysis cases:

```bash
hf download xbb083/BrainBench \
  --repo-type dataset \
  --include "foundational_analysis/**" \
  --local-dir ./benchmarks
```

For a different individual subset, replace `foundational_analysis` with `sleep_assessment`, `neurocognitive_assessment`, or `physiological_integration`. The downloaded files are placed under `benchmarks/<subset>/cases/` and should remain unchanged.

Raw EEG/PSG data must be obtained separately. See the [Dataset Access Guide](DATASET_GUIDE.md) for official access pages, the records used by BrainBench, and the expected dataset folder names.

### 5. Prepare data

First, obtain the required raw datasets from the official sources listed in the [Dataset Access Guide](DATASET_GUIDE.md). For Foundational Analysis, place the five dataset folders directly under one user-selected `<data-root>`:

```text
<data-root>/
├── isruc/
├── bcic2020-3/
├── MentalArithmetic/
├── mumtaz/
└── seedv/
```
After the raw data is organized, prepare Foundational Analysis with:

```bash
python main.py prepare foundational_analysis \
  --data-root /path/to/data-root
```

To prepare a different subset, replace `foundational_analysis` with `sleep_assessment`, `neurocognitive_assessment`, or `physiological_integration`, and pass the corresponding subset's `<data-root>`. Prepared data is written to the output directory configured for the selected subset; Foundational Analysis, for example, is written to `data/core/`.

### 6. Build the CodeAct Docker image

CodeAct executes model-generated analysis code. With `BRAINBENCH_CODEACT_MODE=docker`, the code runs inside an isolated container built from the project image; this is the recommended mode for safer execution and reproducible dependencies. With `BRAINBENCH_CODEACT_MODE=local`, the code runs directly on the host machine without container isolation, so the host must provide the required packages and has a weaker safety boundary.

Build the image:

```bash
docker build -t brainbench-codeact:latest docker/codeact
```

To use Docker mode, set the following variables in `.env`:

```dotenv
BRAINBENCH_CODEACT_MODE=docker
BRAINBENCH_DOCKER_IMAGE=brainbench-codeact:latest
```


Use `BRAINBENCH_CODEACT_MODE=local` only when Docker is unavailable or direct host execution is intended.

### 7. Run the benchmark

```bash
python main.py run foundational_analysis --agent codeact
```

This command runs the Foundational Analysis subset with the built-in CodeAct agent. To run another subset, replace `foundational_analysis` with `sleep_assessment` (Sleep Assessment), `neurocognitive_assessment` (Neurocognitive/Emotion Assessment), or `physiological_integration` (multimodal Physiological Integration). Results are written to `runs/<subset>.json`.

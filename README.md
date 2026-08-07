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
│   │   └── cases/               # reserved for the corresponding release
│   └── physiological_integration/
│       └── cases/               # reserved for the corresponding release
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

<p>Run these commands from the repository root. The current release supports <code>foundational_analysis</code> and <code>sleep_assessment</code>.</p>

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
NEUROBENCH_API_KEY=YOUR_API_KEY
NEUROBENCH_BASE_URL=https://YOUR_PROVIDER_BASE_URL/v1
NEUROBENCH_MODEL=YOUR_MODEL_NAME
```

### 3. Smoke test

```bash
python main.py smoke
```

### 4. Download benchmark cases

The fixed case JSON files are published in the [BrainBench Hugging Face dataset](https://huggingface.co/datasets/xbb083/BrainBench). They contain the benchmark inputs, parsing instructions, ground truth, and metrics; raw EEG/PSG recordings are not included. The current release contains 950 Foundational Analysis instances and 1,025 Sleep Assessment instances.

Download all released cases:

```bash
python -m pip install --upgrade huggingface_hub
hf download xbb083/BrainBench \
  --repo-type dataset \
  --local-dir ./benchmarks
```

To download one subset only:

```bash
SUBSET=foundational_analysis
hf download xbb083/BrainBench \
  --repo-type dataset \
  --include "${SUBSET}/**" \
  --local-dir ./benchmarks
```

The cases are placed under `benchmarks/<subset>/cases/` and should remain unchanged. Download the required raw EEG/PSG data separately; see the [Dataset Access Guide](DATASET_GUIDE.md) for official access pages, the exact records used by BrainBench, and the required local folder layout.

### 5. Prepare data

After obtaining the licensed source data, place it under `downloads/raw/<subset>/original/` as described in the dataset guide. Choose the subset to prepare and run:

```bash
SUBSET=foundational_analysis  # or sleep_assessment
python main.py prepare "$SUBSET" \
  --data-root "./downloads/raw/$SUBSET"
```

Prepared inputs are written to `data/core/` for Foundational Analysis and `data/sleep/` for Sleep Assessment.

### 6. Build the CodeAct Docker image

CodeAct executes model-generated analysis code. With `BRAINBENCH_CODEACT_MODE=docker`, the code runs inside an isolated container built from the project image; this is the recommended mode for safer execution and reproducible dependencies. With `BRAINBENCH_CODEACT_MODE=local`, the code runs directly on the host machine without container isolation, so the host must provide the required packages and has a weaker safety boundary.

```bash
docker build -t brainbench-codeact:latest docker/codeact
```

Keep `BRAINBENCH_DOCKER_IMAGE=brainbench-codeact:latest` when using Docker mode. Switch the mode to `local` in `.env` only when Docker is unavailable or direct host execution is intended.

### 7. Run the benchmark

```bash
python main.py run "$SUBSET" --agent codeact
```

Set `SUBSET=sleep_assessment` to run the Sleep Assessment subset. Results are written to `runs/<subset>.json`. The `neurocognitive_assessment` and `physiological_integration` subsets are reserved for future releases.

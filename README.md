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

Set the model credentials and CodeAct mode in `.env`:

```dotenv
NEUROBENCH_API_KEY=YOUR_API_KEY
NEUROBENCH_BASE_URL=https://YOUR_PROVIDER_BASE_URL/v1
NEUROBENCH_MODEL=YOUR_MODEL_NAME
NEUROBENCH_API_PROTOCOL=openai
NEUROBENCH_REQUEST_TIMEOUT_SEC=300
BRAINBENCH_CODEACT_MODE=docker
BRAINBENCH_DOCKER_IMAGE=brainbench-codeact:latest
```

### 3. Smoke test

```bash
python main.py smoke
```

### 4. Download benchmark cases

```bash
python -m pip install --upgrade huggingface_hub
HF_DATASET_ID="YOUR_ORG/BrainBench-Cases"
SUBSET=foundational_analysis
hf download "$HF_DATASET_ID" \
  --repo-type dataset \
  --include "${SUBSET}/**" \
  --local-dir ./benchmarks
```

The case JSON files should be placed under `benchmarks/<subset>/cases/`. Obtain the licensed raw EEG/PSG data separately and place it under `downloads/raw/<subset>/original/`.

### 5. Prepare, build, and run

```bash
python main.py prepare "$SUBSET" --data-root "./downloads/raw/$SUBSET"
docker build -t brainbench-codeact:latest docker/codeact
python main.py run "$SUBSET" --agent codeact
```

Prepared data are written to `data/core/` for Foundational Analysis and `data/sleep/` for Sleep Assessment. Results are saved to `runs/<subset>.json`.

> `neurocognitive_assessment` and `physiological_integration` are reserved for future releases and are not runnable in the current version.
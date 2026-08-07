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
  <a href="#">&#128640; Quickstart</a>
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
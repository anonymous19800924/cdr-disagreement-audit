# Collaborative Disagreement Resolution

This repository contains the implementation for **Collaborative Disagreement Resolution for Scalable Oversight**. The paper studies an alternative to adversarial AI debate: instead of asking two models to defend fixed opposing answers, Disagreement Resolution (DR) asks consultants to compare reasoning traces, identify concrete conflicts, update their positions, and either converge on a shared answer or expose the remaining crux for a weaker judge.

![Protocol comparison](dr_plot_new.png)

## Repository Layout

```text
.
|-- run.py                 # Main Disagreement Resolution pipeline
|-- run_ablation.py        # DR pipeline with BoN and sycophancy ablations
|-- dr_plot_new.png        # Teaser figure for the README
|-- data/                  # Precomputed consultant traces by dataset/model
|-- prompts/               # DR prompts for reasoning, consultant updates, and judging
|-- ablation/              # Ablation prompts
`-- baselines/             # Naive judge and double-consultancy baselines
```

The checked-in data files contain aligned consultant reasoning traces. The main pipeline matches cases by `index`, initializes each consultant from its precomputed answer and reasoning, and runs the collaborative update loop for up to `--max-turns` turns.

## Setup

Python 3.9 or newer is required. Install dependencies from `pyproject.toml` with Poetry:

```bash
pip install poetry
poetry install
```

Initialize external baseline code:

```bash
git submodule update --init --recursive
```

Run commands through Poetry's environment:

```bash
poetry run python run.py --limit 2
```

The main pipeline depends on `requests`. The baseline scripts additionally import `python-dotenv`, `tenacity`, and `tqdm`; install them if you plan to run `baselines/naive_judge.py` or `baselines/double_consultancy.py`.

## Configuration

`run.py` and `run_ablation.py` read configuration from environment variables or from a root `.env` file. At minimum, set:

```bash
OPENROUTER_API_KEY=your_openrouter_key
DATASET=GPQA # GPQA / SuperGPQA / HLE
CONSULTANT_1=openai/gpt-4o # openrouter model name
CONSULTANT_1_FILENAME=gpt_4o.json
CONSULTANT_2=anthropic/claude-sonnet-4
CONSULTANT_2_FILENAME=claude.json # openrouter model name
JUDGE=openai/gpt-4o-mini # openrouter model name
PROMPTS_SET=prompts
RESULTS_DIR=logs
```

`DATASET` must match a folder under `data/`, and each consultant filename must be a file inside that dataset folder. `PROMPTS_SET` or `PROMPTS_DIR` should point to a prompt directory containing:

- `reasoning_generation.txt`
- `consultant.txt`
- `judge.txt`
- `judge_system.txt`

## Running Disagreement Resolution

Use `run.py` as the general entrypoint for DR experiments. The dataset, models, prompts, and output folder are selected through the environment configuration above, so the same command pattern works for GPQA, SuperGPQA, and HLE.

For a small prompt/debug run:

```bash
python run.py --limit 2 --log-prompts
```

For a specific case:

```bash
python run.py --case-index 0 --log-prompts
```

To resume after API or formatting errors, use cached mode. Cached mode reloads the current output file and retries entries whose `status` is `error`.

```bash
python run.py --cached
```

For rate-limited models, add a delay between API calls:

```bash
python run.py --sleep 10
```

The default number of consultant revision turns is 5:

```bash
python run.py --max-turns 5
```

For GPQA, set `DATASET=GPQA` and choose consultant files from `data/GPQA/`, then call the same runner:

```bash
python run.py --cached
```

## Running Ablations

`run_ablation.py` extends the main DR pipeline with:

- Best-of-N consultant sampling, controlled by `CONSULTANT_1_BON` and `CONSULTANT_2_BON`.
- Sycophancy or anti-sycophancy prompt injections, controlled by consultant-specific sycophancy mode and prompt variables.
- Preference selection through `ablation/BoN/preference.txt` when BoN sampling is enabled.

Example:

```bash
python run_ablation.py --limit 2 --log-prompts
```

## Baselines

Baseline methods live under `baselines/`.

- `naive_judge.py`: asks the judge to answer using only the question and choices, with no consultant reasoning.
- `double_consultancy.py`: shows the judge both consultants' initial answers and reasoning traces, then asks it to choose the more reliable solution.
- `llm_debate/`: git submodule for the debate baseline from Khan et al. (2024). We use the paper's debater without interaction setting as the standard debate comparison.

Example baseline commands:

```bash
python baselines/naive_judge.py \
  --model1_file data/GPQA/gpt_4o.json \
  --model2_file data/GPQA/claude.json \
  --judge_model openai/gpt-4o-mini

python baselines/double_consultancy.py \
  --model1_file data/GPQA/gpt_4o.json \
  --model2_file data/GPQA/claude.json \
  --judge_model openai/gpt-4o-mini
```

The debate submodule points to the original implementation:

- https://github.com/ucl-dark/llm_debate

Reference:

```bibtex
@InProceedings{pmlr-v235-khan24a,
  title = {Debating with More Persuasive {LLM}s Leads to More Truthful Answers},
  author = {Khan, Akbir and Hughes, John and Valentine, Dan and Ruis, Laura and Sachan, Kshitij and Radhakrishnan, Ansh and Grefenstette, Edward and Bowman, Samuel R. and Rockt\"{a}schel, Tim and Perez, Ethan},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {23662--23733},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/khan24a/khan24a.pdf},
  url = {https://proceedings.mlr.press/v235/khan24a.html},
}
```

## Outputs

By default, DR results are written under:

```text
<RESULTS_DIR>/<DATASET>/<judge>_<consultant_1>_<consultant_2>.json
```

With `--log-prompts`, prompt payloads are saved to:

```text
<RESULTS_DIR>/<DATASET>/prompt_log.json
```

Each result records the case index, consultant states, final answers and reasoning, judge decision, correctness, status, and raw model responses where relevant.

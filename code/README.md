# Harness

Everything needed to turn a prompt into weights and a score. The top-level
[README](../README.md) has the results; this file is the operator's guide.

## Files

| file | role |
|---|---|
| `core.py` | MinAtar 6-channel wrapper (`MinAtar6`), `MLPPolicy`, 1,000-episode `evaluate()` |
| `edits_dsl.py` | parser/applier for the `edits` weight-assignment block |
| `llm.py` | OpenAI-compatible client (DeepSeek / ofox / OpenAI), streaming, cost accounting |
| `run_experiment.py` | orchestrates prompt → reply → parse → weights → eval, one directory per attempt |
| `eval_minatar.py` | score one `.npz` weight file |
| `check_reference.py` | bit-exactness gate against the released artifacts in `../ref/exp5` |
| `report.py` | rebuild a results table from a runs directory |
| `baseline_random.py` | uniform-random baseline under the same protocol |
| `prompts/*.md` | the three prompts, verbatim from the released logs |
| `run_all.sh` | gate → 3 games × 3 attempts → table |

## Environment

```bash
conda create -n astra python=3.10 -y && conda activate astra
pip install -r requirements.txt
export MPLCONFIGDIR=$PWD/.mplcache     # MinAtar imports matplotlib
```

## Configuration

The client reads environment variables (or a local `.env`, which is
git-ignored; never commit keys):

| variable | meaning |
|---|---|
| `OFOX_API_KEY`, `OFOX_BASE_URL` | relay endpoint, default `https://api.ofox.io/v1`; model from `LLM_MODEL` |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | DeepSeek endpoint, default `https://api.deepseek.com`; model from `DEEPSEEK_MODEL` (default `deepseek-v4-pro`) |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | any other OpenAI-compatible endpoint; model from `OPENAI_MODEL` |
| `LLM_MODEL` | model id for the relay path (`OFOX_*`) |
| `LLM_REASONING_EFFORT` | `low` / `medium` / `high` / `xhigh` (provider-dependent; `max` is DeepSeek-only) |
| `LLM_TEMPERATURE` | sampling temperature, default 1.0 |
| `LLM_MAX_TOKENS` | completion budget (reasoning tokens count against it), default 65,536 |
| `LLM_STREAM=1` | stream responses; avoids buffering on long reasoning calls |

Provider precedence is ofox → DeepSeek → OpenAI.

## Running

```bash
# sanity gate, no API key required
python check_reference.py --episodes 100        # must print ALL CHECKS PASS

# one attempt per game into runs/local
python run_experiment.py --game all --attempts 1 --episodes 1000 --runs-root runs/local

# one model, three attempts, resuming after a crash
python run_experiment.py --game breakout --attempts 3 --episodes 1000 \
    --runs-root runs/astra --resume

# score a stored weight file, and rebuild a table
python eval_minatar.py --weights ../runs/astra/breakout/attempt1/weights.npz \
    --game breakout --episodes 1000
python report.py --runs-root ../runs/astra --out ../docs/results-astra.md
```

Each attempt directory holds `config.json`, `prompt.txt`, `reply.md`,
`usage.json` (tokens and cost), `parse.json` (statements applied, per-tensor
nonzero counts, parse errors) and `eval.json` (score, SE, max, mean length).

`--dry-run` prints the prompt sizes and the resolved request config without
sending anything; `--mock-reply <file>` replays a stored reply through the parser
and evaluator (useful offline).

# WEDGE: Weakness-Driven Environment Generation for Embodied Navigation Agents

Code and data for the paper *WEDGE* (arXiv, 2026): the **LLM-assisted weakness
discovery** loop and the episode corpora it was run on for the three agents in the
paper.

## Layout

```
weakness_discovery/   discover.py (main loop), gates.py, sandbox.py, llm_client.py,
                      summarize.py (episode -> prompt line), build_dataset.py,
                      compare_to_baseline.py (did the loop rediscover hand-written features?)
data/                 episode corpora, one record per evaluated episode
```

### Corpora (`data/`)

| file | agent | episodes |
|---|---|---|
| `ddppo_pretrained_1550.json` | DD-PPO, ProcTHOR val, 16 categories | 1,550 |
| `ddppo_after_round1_1550.json` | DD-PPO after one WEDGE round | 1,550 |
| `uninavid_pretrained_655.json` | Uni-NaVid, ProcTHOR val, 6 HM3D categories | 655 |
| `uninavid_after_round1_655.json` | Uni-NaVid after one WEDGE round | 655 |
| `gemini_zero_shot_300.json` | Gemini agent, ProcTHOR val subset | 300 |
| `gemini_after_round1_300.json` | Gemini agent after one WEDGE round | 300 |

Each record has `success`, `targetCategory`, `scene` (house and target-room
geometry), `planner` (shortest-path descriptors), `baseline` (six hand-written
features), and `agent` (rollout statistics, context only; empty for the Gemini
agent). The `scene` and `planner` blocks are agent-independent.

## Running it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...          # gemini-2.5-pro via the OpenAI-compatible endpoint
cd weakness_discovery

# from-scratch discovery on the pretrained Uni-NaVid corpus
python discover.py --episodes ../data/uninavid_pretrained_655.json \
    --from-scratch --rounds 5 --tag demo --out-dir ../runs/demo --save-prompts

# another agent: change the corpus, the tag and the one-line agent description
python discover.py --episodes ../data/ddppo_pretrained_1550.json \
    --from-scratch --rounds 5 --tag ddppo --out-dir ../runs/ddppo \
    --agent-desc "an RL ObjectNav policy (DD-PPO, RGB only)"

# plumbing test without an API key
python discover.py --episodes ../data/uninavid_pretrained_655.json \
    --from-scratch --rounds 1 --tag mocktest --out-dir /tmp/wedge_mock --mock mock_responses

# did a from-scratch run rediscover the hand-written features?
python compare_to_baseline.py --episodes ../data/uninavid_pretrained_655.json \
    --ledger ../runs/demo/ledger_demo.json
```

Each run writes `ledger_<tag>.json` (every candidate with its gate scores and
rejection reason), `feature_lib_<tag>.py` (accepted features, importable),
`weakness_table_<tag>.md` (tertile success rates per feature) and
`report_<tag>.md`.

Notes on reproducibility: the LLM is sampled at temperature 0.8, so a rerun
proposes different candidates; the gates are deterministic (fixed seeds).

## Citation

```bibtex
@article{zhang2026wedge,
  title   = {WEDGE: Weakness-Driven Environment Generation for Embodied Navigation Agents},
  author  = {Zhang, Yue and others},
  journal = {arXiv preprint},
  year    = {2026}
}
```

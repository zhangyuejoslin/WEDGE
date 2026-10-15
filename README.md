# From Failures to Learnable Worlds: Policy-Conditioned Environment Generation for Navigation Agents

Yue Zhang, Han Lin, Zun Wang, Yidong Huang, Zaid Khan, Mohit Bansal

UNC Chapel Hill

[Paper (arXiv:2610.19420)](https://arxiv.org/abs/2610.19420) · [Project page](https://zhangyuejoslin.github.io/WEDGE/)

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

## Citation

```bibtex
@article{zhang2026wedge,
  title   = {From Failures to Learnable Worlds: Policy-Conditioned Environment Generation for Navigation Agents},
  author  = {Zhang, Yue and Lin, Han and Wang, Zun and Huang, Yidong and Khan, Zaid and Bansal, Mohit},
  journal = {arXiv preprint arXiv:2610.19420},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.19420}
}
```

# Stage A2 — PC-inspired auxiliary losses on the residual adapter

A2 keeps the A1 setup (a small trainable residual on the hidden states of a frozen language model) and adds **auxiliary losses on the hidden states**. Training is still plain backpropagation with one optimiser: there is **no settling** in A2, so "PC-inspired" describes the shape of the losses, not the way the weights are trained. FabricPC supplies the nodes where it applies (the contrastive heads and the squared-error energies).

Everything stage-level is in one file, [`a2.py`](a2.py). The pure loss math is in [`residual/`](../../residual/).

> **Status.** Built and tested: the objective wiring, the contrastive future loss and the stability loss (anchor and HEP nudge), the runner, ablation arms, and three internal metrics. **Not built yet** (other PRs): the hidden-prediction loss `L_hid`, `L_step`, `L_res-smooth`, and the post-hoc analysis scripts. See [Not implemented yet](#not-implemented-yet).

---

## Contents

- [The objective](#the-objective)
- [The loss terms](#the-loss-terms)
- [Where FabricPC is used](#where-fabricpc-is-used)
- [Parameters](#parameters)
- [Configuration](#configuration)
- [Ablation arms](#ablation-arms)
- [Running](#running)
- [What is logged](#what-is-logged)
- [Internal metrics](#internal-metrics)
- [Code layout](#code-layout)
- [Tests and guarantees](#tests-and-guarantees)
- [Design notes and caveats](#design-notes-and-caveats)
- [Adding a loss](#adding-a-loss)
- [Not implemented yet](#not-implemented-yet)

---

## The objective

$$\mathcal{L}_{A2} = \underbrace{\mathcal{L}_{task} + \lambda_{kl}\,\mathrm{KL}(p_{F_0}\,\|\,p_{F_0+R}) + \lambda_{wd}\lVert\phi\rVert^2}_{\text{A1, unchanged}} + \lambda_{cp}\,\mathcal{L}_{cpc} + \lambda_{stab}\,\mathcal{L}_{stab}$$

$$\mathcal{L}_{stab} = w_{anch}\,\mathcal{L}_{zanch} + w_{hep}\,\mathcal{L}_{hep\text{-}nudge}$$

With `cpc.weight = stab.weight = 0` the objective, the parameter tree and the gradients are exactly A1's (a test checks it).

```text
input ids ─► frozen model ─────────────────────────────► h^base_l  (stop-gradient), base logits
         │
         └─► frozen model + adapter hook on block l ───► h~_l = h_l + R(h_l), adapted logits
                                                          │
   task CE  +  λ_kl·KL  +  λ_wd·‖φ‖²  +  λ_cp·L_cpc  +  λ_stab·L_stab
                                  │
                      jax.grad over the whole parameter tree ─► optax.adam
```

`h_l` is the block output before the adapter; `h~_l` is what the next block receives. The base forward is run only when a term needs it (KL, a base-state positive, or the anchor).

## The loss terms

| Term | Formula | Trains |
|---|---|---|
| `L_cpc` | InfoNCE: from `h~_{l,t}` predict the stop-gradient code of the base (or corrected) state `k` positions ahead | adapter, `cpcq_<l>`, `cpca_<l>_<k>` |
| `L_zanch` | `‖h~_l − sg(h^base_l)‖²`, normalised by the base norm (`normalize: relative`) | adapter |
| `L_hep-nudge` | `‖α · sg(ε_y) · V‖²` with `ε_y = softmax(logits) − onehot(next token)` | highway `hep_<l>` only |

**Contrastive future loss.** Code `c = q(h~_t)` (parameter-free LayerNorm, then a bias-free linear projection to `dim`), prediction `p = A_{l,k} c` (bias-free linear, initialised near identity), both L2-normalised. The positive is `q(h_{t+k})` under stop-gradient, taken from the base state (`positive: base`) or the corrected state (`positive: corrected`). Negatives are the positives of the **other sequences** in the batch. Anchors are subsampled by a deterministic stride (`max_anchors_per_seq`). Padding is masked: an anchor counts only if both `t` and `t+k` are real tokens and at least one real negative exists. Several offsets (`offsets: [1, 2, 4]`) each get their own predictor; the term is the mean over sites `(l, k)`.

**Anchor.** A squared distance between the corrected and the base state, averaged over real tokens and sites. It is 0 at initialisation (`B = 0`) and pulls the adapter back toward the base model.

**HEP nudge.** The output error `ε_y` is a training-time-only signal that is projected into the hidden space by a "highway" matrix `V`. Because `ε_y` is under stop-gradient, this loss reaches **only `V`**, never the adapter. With a trainable highway (`dense`, `lowrank`) it trains `V`; with a fixed one (`fixed_random`) it has no gradient and is a **logged metric** (`hep_nudge_norm`, `hep_nudge_ratio`), so the config rejects `w_hep > 0` for fixed highways.

## Where FabricPC is used

Project policy: use FabricPC wherever it can apply. In A2 it is used for **feed-forward values and Gaussian energies only**:

| Piece | FabricPC node | File |
|---|---|---|
| CPC projection `q` and predictor `A_{l,k}` | bias-free `Linear` | `residual/cpc_heads.py` |
| Anchor squared distance | `IdentityNode` energy, base state clamped as the target | `residual/fpc_energy.py` |
| HEP squared norm of the nudge | `Linear` node(s) (low-rank through a `mid` node), zero target | `residual/fpc_energy.py` |

This is backprop **through** FabricPC nodes, not predictive-coding training (no inference loop). The LayerNorm and the InfoNCE (which needs the other sequences as negatives, so it is not a per-sample energy) stay in JAX. A plain-JAX reference backend (`heads: jax`, `backend: jax`) is kept as the test oracle: values, gradients and `jit` agree with the FabricPC path.

## Parameters

One flat, two-level tree, the same shape as A1's, so `run_sequential`, checkpoints (`RunWriter.save_params`) and `evaluate` work unchanged:

| Key | Leaves | Present when |
|---|---|---|
| `layer_<l>` | `A [d,r]`, `B [r,d]` | always (the adapter) |
| `cpcq_<l>` | `W [d,dim]` | `cpc.weight > 0` |
| `cpca_<l>_<k>` | `M [dim,dim]` | `cpc.weight > 0`, one per offset |
| `hep_<l>` | `V [vocab,d]` (`dense`) or `U, W` (`lowrank`) | `stab.weight > 0` and a trainable `hep.kind` |

`init_a2_params(config, adapter)` builds it; with every auxiliary term off it equals `adapter.init_params()`. `adapter_params(tree)` returns the `layer_<l>` sub-tree; the weight decay `λ_wd‖φ‖²` and `ResidualAdapter.num_params` use **only** that sub-tree. `param_counts(tree)` reports `adapter`, `cpc`, `hep` and `total`. Head initialisation uses its own seeded stream, so it never collides with the adapter's.

## Configuration

The λ's and every loss setting are in the `objective:` section of [`configs/stage_A/a2_pc_inspired.yaml`](../../configs/stage_A/a2_pc_inspired.yaml); the dataclasses are in `a2.py`. Unknown keys are rejected at every level. `adapter`, `training` and `domains` are identical to A1's YAML (a test checks it), so an A2 run differs from A1 only in `objective`.

```yaml
objective:                  # → A2Config
  lambda_kl: 0.0            # A1 terms
  lambda_wd: 0.0
  seed: 0                   # seed of the heads and of a fixed-random highway
  cpc:                      # → CPCConfig
    weight: 0.05            # λ_cp; 0 switches the term and its heads off
    offsets: [1]            # future offsets k (paper: 1, 2, 4)
    dim: 32                 # width of the CPC code
    tau: 0.1                # temperature
    positive: base          # base | corrected
    max_anchors_per_seq: 256
    a_init_noise: 0.01
    heads: fabricpc         # fabricpc | jax
  stab:                     # → StabilityConfig
    weight: 0.1             # λ_stab; 0 switches the term off
    w_anch: 1.0
    w_hep: 0.0              # > 0 needs hep.kind: lowrank | dense
    normalize: relative     # relative | none
    backend: fabricpc       # fabricpc | jax
    hep:                    # → HEPConfig
      kind: fixed_random    # null | dense | lowrank | fixed_random
      rank: 16
      alpha: 1.0
```

| Setting | Default | Notes |
|---|---|---|
| `cpc.weight` (λ_cp) | 0.0 (YAML: 0.05) | first runs: small |
| `stab.weight` (λ_stab) | 0.0 (YAML: 0.1) | |
| `stab.w_anch` / `stab.w_hep` | 1.0 / 0.0 | |
| `cpc.tau` | 0.1 | 0.05 – 0.2 |
| `cpc.dim` | 32 | 32 or 64 |

The shipped values are first-run guesses and are **not calibrated**. Precedence, weakest to strongest: YAML → `--arm` → `--lambda-*` flags.

`hep.kind: tied` (the frozen LM head as `V`) is supported by the loss function but **rejected by `A2Config`** until `a2_objective` can obtain that matrix.

## Ablation arms

An arm chooses which terms run and their variant; `λ_kl`, `λ_wd`, data, optimiser and seeds are shared, so contrasts against `a1_mode` isolate one term. The λ values come from the config (a weight of 0 falls back to cpc 0.05 / stab 0.1).

| `--arm` | Terms on |
|---|---|
| `a1_mode` | none: the A1 objective |
| `stab_anch` | anchor only |
| `stab_hep` | anchor + trainable low-rank HEP highway |
| `cpc_base` | contrastive future, positive from the base state |
| `cpc_corrected` | contrastive future, positive from the corrected state |
| `cpc_multi` | contrastive future, offsets 1, 2, 4 |
| `full` | `cpc_base` + `stab_anch` |

## Running

```bash
# The shipped configuration
python experiments/run_stage_a2.py --config configs/stage_A/a2_pc_inspired.yaml

# Ablation arms, one seed each (sets adapter, training and objective seeds)
python experiments/run_stage_a2.py --arm a1_mode --seed 0
python experiments/run_stage_a2.py --arm cpc_base --seed 0
python experiments/run_stage_a2.py --arm full --seed 0

# Weights from the command line (override the YAML and the arm)
python experiments/run_stage_a2.py --lambda-cp 0.1 --lambda-stab 0
python experiments/run_stage_a2.py --lambda-kl 0.1 --lambda-wd 1e-4

# Smoke run, and evaluating a saved checkpoint (the auxiliary heads are ignored)
python experiments/run_stage_a2.py --rank 8 --epochs 1 --run-name smoke
python experiments/run_stage_a2.py --eval-checkpoint runs/<run>/params/task_2_scotus.npz
```

Other flags are those of the A1 runner (`--model`, `--rank`, `--epochs`, `--output-dir`, `--run-name`, `--split`, `--eval-batches`). A run writes `runs/<name>/` (default `a2[-<arm>][-s<seed>]-<UTC time>`) with `config.yaml`, `meta.json` (including `arm`, `seed` and the parameter counts by group), `metrics.jsonl`, `results.json` and `params/task_<t>_<name>.npz`. `results.json` has the same external metrics as A1 (`loss`, `ppl`, `kl`, `improvement`, `mean_seen_loss`, `mean_forgetting`, `mean_step_time`, `params_unchanged`).

> The training step of `run_sequential` is not JIT-compiled, so CPU runs are slow. The objective itself is jit-compatible (a test compares jit and eager values and gradients).

## What is logged

`terms` always holds `total` and `task`; every key is logged as `loss_<name>` in `metrics.jsonl` and averaged per epoch.

| Term | Meaning |
|---|---|
| `total`, `task`, `wd`, `kl` | A1 terms (`kl` appears when computed) |
| `cpc`, `stab` | the auxiliary terms **before** their λ |
| `cpc_top1`, `cpc_pos_sim`, `cpc_neg_sim`, `cpc_n_anchors`, `cpc_valid` | contrastive diagnostics (top-1 accuracy of the positive, similarities, counted anchors) |
| `stab_anch`, `anch_dist`, `anch_ratio` | anchor loss, raw distance, relative distance |
| `stab_hep`, `hep_nudge_norm`, `hep_nudge_ratio` | HEP loss and the size of the nudge |

## Internal metrics

Added to [`metrics/representation.py`](../../metrics/representation.py) (pure functions, tested):

| Function | Computes |
|---|---|
| `residual_norm_ratio(h, h_corrected, mask=None)` | mean `‖R‖ / ‖h‖` over tokens |
| `direction_redundancy(vectors)` | mean off-diagonal `|cos|` among the rows |
| `residual_redundancy(A, B)` | `{read, write}`: redundancy of the columns of `A` and the rows of `B` |
| `gradient_conflict(g1, g2)` | `[−cos(g1, g2)]₊` of two gradient PyTrees |

The post-hoc script that applies them to saved checkpoints is not written yet.

## Code layout

| File | Purpose |
|---|---|
| [`stages/stage_A/a2.py`](a2.py) | **Everything stage-level**: configs, parameter tree, `cpc_term`, `stability_term`, `a2_objective`, arms |
| [`residual/aux_losses.py`](../../residual/aux_losses.py) | Pure losses: `contrastive_future_loss`, `anchor_loss`, `output_error`, `hep_nudge_loss` |
| [`residual/cpc_heads.py`](../../residual/cpc_heads.py) | CPC heads `q` and `A_{l,k}` (FabricPC nodes) and their reference |
| [`residual/hep.py`](../../residual/hep.py) | HEP highway matrices (`dense`, `lowrank`, `fixed_random`, `tied`) |
| [`residual/fpc_energy.py`](../../residual/fpc_energy.py) | FabricPC energies for the anchor and the HEP norm |
| [`metrics/representation.py`](../../metrics/representation.py) | Internal metrics (above) |
| [`experiments/run_stage_a2.py`](../../experiments/run_stage_a2.py) | CLI runner (`--arm`, `--seed`, `--lambda-*`) |
| [`configs/stage_A/a2_pc_inspired.yaml`](../../configs/stage_A/a2_pc_inspired.yaml) | Run configuration |

`a2.py` is ordered: 1 configuration, 2 parameters, 3 terms, 4 objective, 5 arms. Each term is a pure function of dictionaries of hidden states `{layer: [B, T, d]}` and never touches the substrate.

## Tests and guarantees

```bash
pytest tests/residual/ -v
```

About 395 A2 tests (plus 20 in `metrics/test_representation.py`); the whole suite is 628 passed, 2 skipped. All run on tiny GPT-2 and GPT-NeoX fixtures on CPU: nothing has been run on a real checkpoint or a GPU.

| File | Covers |
|---|---|
| `stage_A/test_a2_objective.py` | `A2Config` and the YAML, the parameter tree, A1 parity, term weighting, gradient audit, padding, jit vs eager, arms, `run_sequential`, the runner end to end |
| `stage_A/test_a2_cpc.py`, `test_a2_stability.py` | the two terms through the frozen model (GPT-2 and NeoX), configs, reduction over sites |
| `test_aux_losses.py`, `test_contrastive_mask_backend.py` | InfoNCE, anchor, output error, HEP nudge, padding mask, backend switch |
| `test_cpc_heads*.py`, `test_stability_*.py`, `test_hep.py` | heads, energies and highways; FabricPC against the plain-JAX reference |

Guarantees enforced by tests:

- **A1 parity**: with the auxiliary weights at 0, total, terms and gradients equal `a1_objective`'s, and a short sequential run matches A1's.
- **Gradient audit**: `L_hep-nudge` reaches only the highway; the anchor and `L_cpc` reach the adapter; each term enters the gradient linearly in its weight; the weight decay covers the adapter only.
- **FabricPC = reference**: values, gradients and `jit` agree to about 1e-5; spy tests prove the `fabricpc` setting really runs through FabricPC nodes.
- **Padding**: padded tokens have no influence on the masked terms and get no gradient.
- **Frozen base**: base weights are unchanged after every objective call and run.
- **Mutation checks** (run by hand, not in CI): planted bugs in the objective, the terms and the heads were caught, apart from changes that alter nothing observable.

## Design notes and caveats

- **No settling.** The nodes are used as differentiable layers; L_step and the settled representation are for a later stage.
- **Mask source.** The A1 objective receives no attention mask, so the padding mask is `labels != -100`. Tokens whose label is ignored for the task loss are also left out of the contrastive and anchor terms. With no `labels`, every token counts.
- **Where `h^base` comes from.** The adapted and base runs are separate forwards; the base states are the output of the unmodified model, not of the adapted model's earlier layers.
- **Logged names.** The shared loop prefixes every term with `loss_`, so diagnostics appear as `loss_cpc_top1`.
- **Cost not measured.** FabricPC builds a small graph state on every head call; it has not been benchmarked against a plain matrix product, and `mean_step_time` per arm has not been recorded.
- **Hyperparameters are uncalibrated.** A calibration of each term's gradient norm against the task gradient is planned, not written.
- **Dependency.** `fabricpc==0.6.0` (PyPI) needs `jax>=0.7.0`; it is the only version tested.

## Adding a loss

Teammates adding `L_hid`, `L_step` or `L_res-smooth` follow the same three steps:

1. **Config**: a frozen dataclass section (`weight` plus its settings, `from_dict` through `checked_kwargs`), added as a field of `A2Config`.
2. **Term**: a `*_term(params, *, config, corrected, base, ...)` function returning `(unweighted_loss, metrics)`; put the pure math in `residual/` and use FabricPC nodes where they apply. Head parameters get their own key prefix and an `init_*` function called by `init_a2_params`.
3. **Objective**: a few lines in `a2_objective` that call the term, add `weight * loss` to `total`, and put the unweighted loss and its metrics in `terms`. Add the arm(s) to `ARMS` / `apply_arm`.

The runner, the checkpoints and `evaluate` need no change. Add tests next to the existing ones: config validation, parameter keys, term value against a plain reference, gradient audit, A1 parity at weight 0.

## Not implemented yet

- `L_hid`, the hidden-prediction loss (`P_l` with FabricPC nodes), and its arms.
- `L_step`, `L_res-smooth` (a later stage / other PRs).
- `hep.kind: tied` in `a2_objective`.
- `experiments/analyze_stage_a2.py` (internal metrics from checkpoints, paired statistics with Holm correction) and `experiments/calibrate_stage_a2.py` (λ calibration).
- A JIT-compiled training step in `run_sequential` (needed before large sweeps).
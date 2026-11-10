# A Subset Indexed Attention Cache for Exact Attention over Many Subsets

Release repository for the paper of that title. It holds the library source at the versions measured, the five notebooks that produced the paper's measurements and generated tables, the saved result files, the 18 benchmark inputs, the four pilot data extracts and a manifest of file hashes.

Repository: https://github.com/jakemdrew/subset_cache_pub
Release tag and commit: to be recorded at submission.

## Module hashes

Every library module is identified by the full SHA256 of its file. The pilot notebook recorded all nine before training, in pilot/runs/v3_gate.pkl.lz4. The two benchmark result files recorded the five the benchmarks import. The manifest notebook verifies every module on disk against the gate record and writes the full digests to MANIFEST.md. The first 16 hexadecimal digits are listed here.

| File | SHA256, first 16 | Role |
|---|---|---|
| src/set_attention_cache.py | 86d8f308bbbc1a79 | the sorted order, unranking, the successor rule, the trie builder, the run trie generator and the three samplers |
| src/set_attention_model.py | d7a7e72a2a111bb4 | the cache, implementation (a), level walk forward and closed form backward pass |
| src/set_attention_data.py | fd4c5a486477ccc3 | level plans and device transfer |
| src/set_attention_bench.py | 80b9a7eddb88ddee | the benchmark harness |
| src/set_attention.py | 8580eb3ce583edec | shared definitions |
| src/masked_concept_model.py | f6b49be2e07791e1 | the pilot model |
| src/mc_batched.py | 571185cc24b467ee | the pilot's batched loss over the cache |
| src/mc_pilot_data.py | 8bddded271636eda | the pilot data extraction and microbatch plans |
| src/mc_walk_jump_custom.py | 385bfec34ad32cb8 | pointer jumping under autograd, implementation (c), the pilot reference |

Also shipped: src/apriori/serialization.py and an empty src/apriori/__init__.py, which resolve the notebooks' apriori.serialization import when src/ is on the Python path.

## Layout

| Folder | Contents |
|---|---|
| src/ | the library and pilot source listed above |
| notebooks/ | V3 Benchmarks 2080.ipynb, V3 Benchmarks H100.ipynb, V3 Pilot 2080.ipynb, V3_Benchmarks_Tables.ipynb, V3 Release Manifest.ipynb |
| inputs/ | the 18 frozen benchmark inputs, generated once from recorded seeds and read by both machines, each verified by hash before use |
| results/ | section7_benchmarks_2080.pkl.lz4 and h100_v3/section7_benchmarks_H100.pkl.lz4, the two benchmark result files |
| results/tables/ | the nine benchmark tables, paragraph.tex, fig1_speedup.png, derived.pkl.lz4 and derived_best_mode.pkl.lz4, written by the tables notebook |
| pilot/ | the four pilot data extracts, mc3_pilot_data_d5.pkl.lz4, mc3_contexts_d5.pkl.lz4, mc3_decile_edges.pkl.lz4, mc3_pilot_data_d5_removed.pkl.lz4. The contexts file is a contract input. The pilot notebook loads it and hashes it into every contract, and it is not used to select contexts |
| pilot/runs/ | the three runs W0_registered, W0 and W, each with v3_{run}_curve.pkl.lz4 and the final checkpoint v3_{run}_last.pt, and the files v3_gate.pkl.lz4, v3_pilot_report.pkl.lz4 and v3_cost_study.pkl.lz4 written by the pilot notebook's Cells 4, 7 and 8 |
| pilot/tables/ | tab_pilot_baseline.tex, tab_pilot_endpoints.tex and tab_pilot_bins_sampled.tex, written by the pilot notebook's Cell 9 |
| MANIFEST.md | every file above with its full SHA256, size and role, written by the manifest notebook |

MANIFEST.md lists every file of the release other than itself. All data files are pickles compressed with lz4 (.pkl.lz4). The H100 results file was written by the pod's pandas and is read through the compatibility loader in the tables notebook's Cell 1 and the manifest notebook's Cell 1. Every other file reads through apriori.serialization.load_from_pickle.

## Notebooks and cells

Every notebook runs top to bottom in a fresh kernel. The benchmark and pilot notebooks recompute the library module hashes before any measurement and stop if they differ from the recorded ones. The tables notebook imports no library module. The manifest notebook imports none in Cell 1 and imports set_attention_cache in Cell 2. Run times marked estimate were not timed in the release execution.

### V3 Benchmarks 2080.ipynb and V3 Benchmarks H100.ipynb

The two notebooks hold the same cells. MACHINE in Cell 1 is the only configuration difference. The H100 notebook is the copy uploaded to the pod. Its executed copy was not retrieved before the pod was terminated, so the shipped file carries no outputs. The H100 results file is the record of that run, with the machine environment, the module hashes, the schema version and the suite timing in its metadata.

| Cell | Purpose | Run time |
|---|---|---|
| 1 | Configuration, implementation identity, environment record and the benchmark manifest | seconds |
| 2 | Verify every input file present by content hash. An attention input whose file is missing is regenerated from its recorded seed. An existing file is never replaced | under 3 minutes with all 18 files present, 20 to 40 minutes to regenerate the 14 attention inputs |
| 3 | Generate the 4 training step benchmark inputs from the source data store | not reproducible without the source data store, the files are shipped |
| 4 | Harness, definitions only | under 5 seconds |
| 5 | The suite, resumable, saves after every evaluation | 2080 1.61 h operational, H100 1.50 h, each 18 benchmarks, 306 evaluations, 288 agreement checks and 360 preparation measurements |
| 6 | Run trie generator equivalence. Part A, 25,530 enumerated cases. Part B, the 14 run sampler and full power set inputs, where the intervals recomputed from each input's key must reproduce the stored subsets in order and the generated trie must equal the built trie | about 5 minutes on the 2080 |
| 7 | Preparation components on the two largest rows, timed in isolation | about 15 minutes on the 2080 |
| 8 | Preparation per method from the store | under 10 seconds |

### V3 Pilot 2080.ipynb

The three runs were trained by this notebook's Cell 6 on 2026-11-09, in the execution whose outputs it carries. The gate record was written before training, at 14:15, and the first checkpoint at 14:51. The report, cost study and tables were written by Cells 7 to 9 on 2026-11-10, and Cell 10 was added afterwards.

| Cell | Purpose | Run time |
|---|---|---|
| 1 | Test of the cache's log normalizer output against an independent float64 reference. Fresh kernel, and the kernel is restarted before Cell 2 | under 1 minute |
| 2 | Configuration, library identity and data provenance. Prints the nine module hashes, the environment, the data fingerprints and the counts | 1 to 3 minutes |
| 3 | Agreement gate, the cache against the reference traversal, pointer jumping under autograd, on the first accumulation group of epoch 1 | 1 to 3 minutes |
| 4 | Gate record, writes pilot/runs/v3_gate.pkl.lz4 with the module hashes, environment, tolerances, settings and gate tables | seconds |
| 5 | Training harness for the three runs, epoch plans, held out set checks and the three contracts | 1 to 3 minutes |
| 6 | Training, the three runs. With the shipped checkpoints complete, each run is loaded under its contract, verified and skipped. A missing or incomplete checkpoint makes the cell train that run from initialization | about 2 minutes to verify, estimate. The recorded training, two epochs each, took 3.77 h for W, 0.69 h for W0 and 1.12 h for W0_registered, 5.6 h in all |
| 7 | Paired intervals, bin results and per target comparisons, writes v3_pilot_report.pkl.lz4 | under 2 minutes |
| 8 | Matched cost study and agreement checks, writes v3_cost_study.pkl.lz4 | 3 to 8 minutes, estimate |
| 9 | Tables for Section 8, writes pilot/tables/, and prints the unrounded hours and ratios of the cost study | seconds |
| 10 | Parameter count from the saved model state of v3_W_last.pt | seconds |

### V3_Benchmarks_Tables.ipynb

| Cell | Purpose | Run time |
|---|---|---|
| 1 | Load the two benchmark result files through the compatibility loader, verify them against each other and combine them | under 10 seconds |
| 2 | Derived fields, the per method preparation charges, the nine tables and the numerical agreement paragraph | under 10 seconds |
| 3 | Figure 1, speedup against budget | under 10 seconds |

### V3 Release Manifest.ipynb

| Cell | Purpose | Run time |
|---|---|---|
| 1 | Hash every release file, verify the nine modules against the gate record, print the module hashes recorded in both benchmark result files, write MANIFEST.md | 1 to 3 minutes |
| 2 | Counting checks of Section 4, CPU enumeration, asserts the theorems and prints the cited tables | 1 to 3 minutes |

## Regeneration map

| Paper item | Regenerated by | From |
|---|---|---|
| Tables of Section 7 that are generated (main attention, main step, boundary, control, ceiling, preparation attention, preparation step, replay setup, preparation plus execution) and Figure 1 | V3_Benchmarks_Tables.ipynb Cells 1 to 3 | the two benchmark result files |
| Numerical agreement paragraph of Section 7 | V3_Benchmarks_Tables.ipynb Cell 2, paragraph.tex | the two benchmark result files |
| Pilot baseline, endpoint and bin tables of Section 8 | V3 Pilot 2080.ipynb Cell 9 | the three curve files and v3_pilot_report.pkl.lz4 |
| Every pilot interval | V3 Pilot 2080.ipynb Cell 7 | the three curve files and the pilot data extracts |
| Pilot agreement checks and cost study of Section 8 | V3 Pilot 2080.ipynb Cell 8 | the shipped checkpoints and the pilot data extracts, on a compatible GPU |
| Benchmark results | V3 Benchmarks 2080.ipynb and V3 Benchmarks H100.ipynb Cell 5 | the 18 inputs, on a compatible GPU |
| Generator equivalence of Section 5 | V3 Benchmarks 2080.ipynb Cell 6 | the 14 run sampler and full power set inputs |
| Counting checks of Section 4 | V3 Release Manifest.ipynb Cell 2 | nothing, CPU enumeration |
| Pilot training runs | V3 Pilot 2080.ipynb Cell 6 | the pilot data extracts, after removing the shipped checkpoints |

The claims table and the reuse table of the paper are written by hand from the same files. The bounds table follows from the algorithms of Section 5, and the implementation map from the source code.

Intervals. Every interval in Section 8 is a record clustered bootstrap. Resample the held out records with replacement 2,000 times under seed (367, 91), take the mean of the paired per target differences in each resample, and report the 2.5 and 97.5 percentiles. The curve files hold the per target losses in held out order. The record identifier of each held out target comes from the pilot data extract, as Cell 5 derives it.

## Environments

| | 2080 | H100 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2080 SUPER, 8,192 MiB, compute capability 7.5 | H100 80GB HBM3, compute capability 9.0 |
| torch | 2.5.1+cu121 | 2.8.0+cu128 |
| CUDA | 12.1 | 12.8 |
| cuDNN | 90100 | 91002 |
| Python | 3.12.7 | 3.12.3 |
| OS | Windows 10 Pro 22H2, build 19045 | Linux kernel 6.8.0, glibc 2.39 |
| TF32 matmul and cuDNN | both recorded False | both recorded False |
| CPU | AMD Ryzen 9 3950X, 16 cores, 128 GB DDR4, from the machine's specification, not the environment record | 23.8 core share of an Intel Xeon Platinum 8480+, 224 visible CPUs, no torch thread limit set, from the pod's lscpu and cgroup cpu.max readings |
| NVIDIA driver | not recorded | not recorded |

The RTX 2080 SUPER predates hardware support for TF32, and the flags are recorded disabled on both machines. The H100 was a rented single GPU cloud instance that no longer exists.

## Recorded hashes

First 16 hexadecimal digits of SHA256 of the file bytes. Full digests are in MANIFEST.md.

| Item | Hash |
|---|---|
| results/section7_benchmarks_2080.pkl.lz4 | 6463ce426816b440 |
| results/h100_v3/section7_benchmarks_H100.pkl.lz4 | 16b5db6768b78910 |
| pilot/runs/v3_W0_registered_curve.pkl.lz4 | 9123c0400f107601 |
| pilot/runs/v3_W0_curve.pkl.lz4 | ac3832644be20cfe |
| pilot/runs/v3_W_curve.pkl.lz4 | b8d04248d2a194b5 |
| pilot/runs/v3_W0_registered_last.pt | 2668f09389e13757 |
| pilot/runs/v3_W0_last.pt | f128c57e1c504a70 |
| pilot/runs/v3_W_last.pt | 60be33797f637b78 |
| pilot/runs/v3_gate.pkl.lz4 | e2da89f20064f75d |
| pilot/runs/v3_pilot_report.pkl.lz4 | 0a285575ee6a8dfd |
| pilot/runs/v3_cost_study.pkl.lz4 | dab57d5e88a2f098 |
| pilot/mc3_pilot_data_d5.pkl.lz4 | aa462904622bd717 |
| pilot/mc3_contexts_d5.pkl.lz4 (contract input) | e5b25a32a6f99946 |
| pilot/mc3_decile_edges.pkl.lz4 | f6270354a9b66881 |
| pilot/mc3_pilot_data_d5_removed.pkl.lz4 | 2281213e46360dd5 |

Pilot identities asserted by the notebook before any run. The three contracts are the values Cell 5 printed and Cell 6 trained under.

| Item | Hash |
|---|---|
| model initialization state | 88071b7a2e52af9f |
| epoch 1 plan | c9ec2fc1ce2e0860 |
| epoch 2 plan | 8aef1402fd7de8d8 |
| pilot values fingerprint | 1d6ee181e9fd1807 |
| contexts fingerprint (contract input) | bb82e1d833c2983b |
| contract, baseline under the registered rule, W0_registered | de1c787361806f25 |
| contract, baseline, W0 | 235932e5fe71eb60 |
| contract, subset supervised model, W | 3e697e75dd41d069 |

A contract is a hash of the module sources, data fingerprints, split, configuration and harness version.

## Pilot settings

Transcribed from masked_concept_model.py, mc_batched.py and the pilot notebook's Cells 2 and 5. In the notebook the baseline and the subset supervised model are the runs W0 and W, and the baseline under the registered rule is W0_registered.

Notation. A stored value v is a percentage change expressed as a fraction (0.25 for 25 percent), and g(v) = sign(v) log(1 + |v|). For a context, either all of the record's other features or a subset of them, c denotes the number of features it contains.

Model. d = 256, H = 4 heads of 64, two encoder blocks, prediction head width 256, Huber delta = 1, 1,647,627 parameters, counted by Cell 10 over the 35 floating point tensors of the saved model state.

- Feature encoder, applied to each feature on its own. The feature's 384 dimensional concept name vector passes through a bias free linear adapter to d. The term g(v) w_val + b_val is added. Two residual blocks follow, each LayerNorm, linear d to 4d, GELU, linear 4d to d. A final LayerNorm comes next, and then linear key and value projections d to d, split into H heads.
- Concept query. The target concept's name vector passes through its own adapter, a learned mask vector is added, and then LayerNorm and a linear projection d to d split into H heads.
- Score z = k^T q / sqrt(64) per head, with k the feature's key.
- Prediction head. Its input is the output vector of each head (H by 64), the per head log normalizer m + log s (H), log(1 + c), and the concept query embedding (d). It applies LayerNorm, linear to 256, GELU, linear 256 to 256, GELU, linear to 1. For a subset, c is formed from the terminal's depth in the trie.

Construction is seeded with 367 and is deterministic.

Optimization.

- AdamW with learning rate 3e-4, weight decay 0.01, betas (0.9, 0.999), eps 1e-8, float32, no gradient clipping. A nonfinite gradient stops the run before the update.
- Per target losses are accumulated over successive microbatches. One update is taken at 32 or more targets, or at the end of the epoch, with gradients divided by the accumulated target count.
- The learning rate is set by cumulative target exposure t, evaluated at the end of each accumulation group. With eta = 3e-4, t_w = 123,490.2 (10 percent of one epoch's 1,234,902 targets) and t_T = 9,879,216 (8 epochs), eta(t) = eta t / t_w for t < t_w, and eta/2 (1 + cos(pi min(t - t_w, t_T - t_w) / (t_T - t_w))) for t at or above t_w. Two of the 8 epochs were run, and the rate at the end of epoch 2 is 2.59e-4.
- Microbatch order within an epoch is a permutation seeded by (367, epoch, 131), and every training occurrence is a target exactly once per epoch.
- Under the accumulated rule, the subset supervised model and the baseline each take 32,995 updates in epoch 1 and 32,958 in epoch 2. The baseline under the registered rule takes one update per microbatch.

Objectives and masking.

- The baseline uses the context term alone. The context is the whole record with the target removed.
- The subset supervised model adds a subset term over subsets of the record's features drawn by the run sampler, with requested budget M = 2,048 and run length lambda = n, keyed by the record name and the epoch. Each target's trie is built from the drawn subsets that exclude the target, so an unusable subset never enters the trie. The usable subset losses are summed and divided by the Horvitz Thompson denominator P (2^(n-1) - 1) / (2^n - 1), computed in float64 as P (1 - 2^-(n-1)) / (2 - 2^-(n-1)) without forming 2^n.
- The per target loss of the subset supervised model is 0.5 times the context term plus 0.5 times the subset term. A target with no usable subset contributes only the 0.5 weighted context term.
- The data loader constructs each microbatch's items, the subset draw and one trie per target built from it, outside the timed execution of the cost study. The run trie generator is not used for these per target tries.

Evaluation.

- Every held out feature is a target. Each held out prediction uses all other features of the record as its context, with the target excluded, for the baseline and the subset supervised model alike. No sampled subset enters evaluation.
- Losses are Huber in signed log space, per target.
- Intervals, 2,000 resamples of held out records with replacement, seed (367, 91). Each interval is the 2.5 and 97.5 percentiles of the resampled mean of the paired per target differences.
- Context permutation probe. 20 held out records with at most 1,024 features, chosen with seed (367, 181). Every feature of each probed record is a target. Target identities, concept names and context membership stay fixed. The features' values are reassigned across the record's positions by one permutation per record, with seed (367, 171, record). Because the permutation is record wide, a target's own original value can move to another position inside its context. The probe value is the median absolute change in prediction, pooled over all probed targets of all 20 records. It compares predictions only and uses no labels.

## Pilot subgroup table

The per bin results of the subset supervised model against the baseline at epochs 1 and 2 are pilot/tables/tab_pilot_bins_sampled.tex, generated by V3 Pilot 2080.ipynb Cell 9 from v3_pilot_report.pkl.lz4.

## Data availability

The pilot data extracts are supplied, so the pilot retrains from this repository. Regenerating the extracts, or the four training step benchmark inputs, requires the source data store, which is not supplied. The 14 attention benchmark inputs regenerate from recorded seeds.

## Running

Put src/ on the Python path. Dependencies are torch with CUDA, triton, numpy, pandas, scipy and lz4. Benchmarks require a CUDA device. On the 8 GB 2080 the unchunked cache ran out of memory at M = 512,000 and fell back to three groups, cache replay ran out of memory at M = 256,000 and above, and the out of memory direct combinations are recorded in the results with that status. Each notebook names its own path constants in its first code cell.

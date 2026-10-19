# Subset attention cache: paper package

Code, notebooks and results behind the paper on exact attention over many subsets of a set through a canonical prefix trie. Every number in the paper is printed by a cell in one of the three notebooks below and saved in a results file listed in `MANIFEST.md`; each notebook is standalone and names the files it reads and writes in its first cell.

## What is claimed, and where it is supported

| Claim | Notebook | Cells |
|---|---|---|
| Agreement within tolerance: the cache's outputs and gradients agree with direct attention within the stated tolerances, in float64 and float32, on both GPUs, through a complete training step | Powerset Validation Cache Benchmarks v3; Powerset Validation Cache Benchmarks H100 | V2C1, V2C4; H1, H4 |
| Sampler validity: the `powerset_runs` sampler gives every nonempty subset the same marginal inclusion probability, checked exhaustively for n = 2 to 8 | v3 | V2C2 |
| Reuse: shared prefix states per summed subset fact approach the exhaustive ratio, as a function of coverage and sampler | v3 | V2C2, V2C3b |
| Speed: faster than the fastest direct configuration for the run sampler on large sets, with the advantage rising with the subset budget for the four run sampler series measured on the H100; slower on small sets, at the smallest budget on the H100, and for samplers that share little | v3; H100 | V2C3a, V2C3b, V2C4; H3, H4, H5 |
| Memory: not lower than the leanest direct configuration on either GPU; lower than the fastest retained direct configuration at 26 of 28 H100 attention only points and all 10 H100 complete step points, because those configurations are unchunked | v3; H100 | V2C3a, V2C4; H3, H4 |
| Application demonstration: the cache trains two subset supervised objectives; at one seed both improve on their baselines at a registered endpoint; at that scale scoped direct enumeration was faster for the sampled design (W) and slower for the exhaustive design (T) | Masked Concept Subset Attention MC2 v2 | MC3ae to MC3ai |

The application results are bounded demonstrations at one seed, not a general accuracy claim. The cache computes the subset objectives; it is not shown to improve accuracy by itself.

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | the three notebooks, each with a record cell at the top listing its claims, cells, result files, environment and module hashes |
| `src/` | the library at the versions the notebooks ran: `set_attention.py` (uncached reference), `set_attention_cache.py`, `set_attention_model.py`, `set_attention_data.py`, `set_attention_bench.py`, and the masked concept pilot modules `masked_concept_model.py`, `mc_batched.py`, `mc_pilot_data.py` |
| `src/apriori/` | `serialization.py` (pickle plus lz4 helpers) and an empty `__init__.py`, which together resolve the notebooks' `apriori.serialization` import when `src/` is on the Python path |
| `results/pvcb/` | benchmark results: `pvcb_results_v2.pkl.lz4` (RTX 2080 SUPER, written by v3), `pvcb_results_h100.pkl.lz4` (H100), `pvcb_results_2080.pkl.lz4` (the 2080 file under the name the H100 notebook reads; byte identical to `pvcb_results_v2`) |
| `results/mc3/` | pilot results: per target held out losses for every arm and epoch, the fixed report bin edges, the direct enumeration benchmark tables |
| `inputs/` | the two real filings with their concept name embeddings used by the complete training step benchmarks, and the pretrained set encoder they load |
| `data/` | fingerprints and counts of the pilot data, which is derived from SEC filings and is not included |
| `MANIFEST.md` | every file in the repo with its source path and sha256 |

## Reproducing

- Get the release: download and unzip `https://github.com/jakemdrew/subset_cache_pub/archive/refs/tags/v1.0.zip`, or clone the repository and check out tag `v1.0`.
- Install: Python 3.12, then `pip install numpy==1.26.4 pandas==2.2.2 lz4 jupyter` and `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`. The H100 notebook's Install cell pins these.
- Working directory and paths: the notebooks were run from the author's machine, and each cell sets absolute path constants at its top (the folders added to `sys.path` and the output folder it reads and writes). Any working directory works once those constants point into the repository: the `sys.path` entries to `src/`, benchmark output folders to `results/pvcb/`, pilot output folders to `results/mc3/`, and the complete step inputs to `inputs/`.
- Benchmarks (v3, H100): run the cells in order. Timing cells require asynchronous CUDA launches; each notebook's setup cell checks this and stops otherwise. Certification cells (V2C1, H1) run before any timing and stop the notebook on a failure. Results are saved after every point and the cells resume where they stopped. V2C5 and H5 read the results files only and can be run alone to regenerate the figures and claim statuses from the saved data.
- Pilot (MC2 v2): the training cells read the cleaned pilot data and contexts, which are not in the repo (fingerprints in `data/`). The per target losses they produced are in `results/mc3/`, so the paired comparisons, the bin reports and the direct enumeration tables can be recomputed from the saved results without training; the record cell lists which cells need the data.
- Intervals: each curve file stores the per target held out losses for every arm and epoch together with each target's filing identifier. A paired interval is the 2.5 and 97.5 percentiles of the mean paired difference over 2,000 resamples of filings with replacement, seed (367, 91), as computed by `cluster_ci` in each MC3 cell.
- Timings depend on the GPU, driver and kernel versions recorded in each notebook's environment cell; agreement results do not.

## Limits stated in the paper

One seed for the application demonstration; filing bootstrap intervals conditional on the trained models; equal updates rather than equal compute between arms; the direct enumeration baseline in the pilot is target by target and chunked; benchmark speedups hold on the measured grid, hardware and implementations and are reported together with the points where the cache is slower.
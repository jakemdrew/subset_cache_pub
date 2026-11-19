# Release manifest

Generated 2026-11-19 13:11. Paths are relative to the release root. SHA256 is the full digest.

## Module hashes

Full SHA256 of each library module as recorded in the pilot gate record before training, and as found on disk.

| File | Recorded SHA256 | On disk | Match |
|---|---|---|---|
| src/set_attention.py | 8580eb3ce583edec900c993d58793e98d2ef4e9c434f0dddc5a58991918f0d52 | 8580eb3ce583edec900c993d58793e98d2ef4e9c434f0dddc5a58991918f0d52 | yes |
| src/set_attention_data.py | fd4c5a486477ccc30a7c202891ffbc579dc02b2bc5ae5988e94cbe6d7ac23cea | fd4c5a486477ccc30a7c202891ffbc579dc02b2bc5ae5988e94cbe6d7ac23cea | yes |
| src/set_attention_model.py | d7a7e72a2a111bb40e45cbc61ed844349e51ed15c7c22749a817893eb8eee345 | d7a7e72a2a111bb40e45cbc61ed844349e51ed15c7c22749a817893eb8eee345 | yes |
| src/set_attention_bench.py | 80b9a7eddb88ddee8948d997808643e476328e915b7ac09847e3ff25f64ea72a | 80b9a7eddb88ddee8948d997808643e476328e915b7ac09847e3ff25f64ea72a | yes |
| src/masked_concept_model.py | f6b49be2e07791e1e8f39729f881e4488a85943f8de35a84ebf83071d751f330 | f6b49be2e07791e1e8f39729f881e4488a85943f8de35a84ebf83071d751f330 | yes |
| src/mc_batched.py | 571185cc24b467ee1955561b19348522548e328ecd7b0aa1767b8e514d78c43e | 571185cc24b467ee1955561b19348522548e328ecd7b0aa1767b8e514d78c43e | yes |
| src/set_attention_cache.py | 86d8f308bbbc1a7934fde24fc65abd90f5e2ef4aeeabff9a5f9795d39db1ccff | 86d8f308bbbc1a7934fde24fc65abd90f5e2ef4aeeabff9a5f9795d39db1ccff | yes |
| src/mc_pilot_data.py | 8bddded271636eda59db5b258ad3ba7607c0fb708e67b3597aa25bc62f0ff525 | 8bddded271636eda59db5b258ad3ba7607c0fb708e67b3597aa25bc62f0ff525 | yes |
| src/mc_walk_jump_custom.py | 385bfec34ad32cb80496c11205aa18ee02131777719dcdf6f03b9867db12a402 | 385bfec34ad32cb80496c11205aa18ee02131777719dcdf6f03b9867db12a402 | yes |

## Files

| Path | Role | Bytes | SHA256 |
|---|---|---|---|
| README.md | release README | 63748 | fc6426806f1eed90b2020e481e53e0cdc9ab3d91d8071ee9314e11c6277701f4 |
| LICENSE.txt | license | 988 | 3123f0562c88cad367db849d2655031f6b7e13d421403fa5b5f71ad8edc39abb |
| .gitignore | git ignore rules | 35 | 41890ca979527e3984a3616488c11cd8634d16d900a8308606d7e062cdd0ed54 |
| .gitattributes | git attributes, every file stored byte for byte | 9 | 8b9c3028d6467ff3be7c793141927841d70f47f52d0590677438f488ca154992 |
| src/apriori/__init__.py | serialization helper | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| src/apriori/serialization.py | serialization helper | 5429 | d3f1036aef823647c63dcb985bcbb99217e3c008bff9d70e80741cb7d9d88707 |
| src/masked_concept_model.py | library and pilot source | 22834 | f6b49be2e07791e1e8f39729f881e4488a85943f8de35a84ebf83071d751f330 |
| src/mc_batched.py | library and pilot source | 9395 | 571185cc24b467ee1955561b19348522548e328ecd7b0aa1767b8e514d78c43e |
| src/mc_pilot_data.py | library and pilot source | 6622 | 8bddded271636eda59db5b258ad3ba7607c0fb708e67b3597aa25bc62f0ff525 |
| src/mc_walk_jump_custom.py | library and pilot source | 8375 | 385bfec34ad32cb80496c11205aa18ee02131777719dcdf6f03b9867db12a402 |
| src/set_attention.py | library and pilot source | 3427 | 8580eb3ce583edec900c993d58793e98d2ef4e9c434f0dddc5a58991918f0d52 |
| src/set_attention_bench.py | library and pilot source | 11013 | 80b9a7eddb88ddee8948d997808643e476328e915b7ac09847e3ff25f64ea72a |
| src/set_attention_cache.py | library and pilot source | 34110 | 86d8f308bbbc1a7934fde24fc65abd90f5e2ef4aeeabff9a5f9795d39db1ccff |
| src/set_attention_data.py | library and pilot source | 13581 | fd4c5a486477ccc30a7c202891ffbc579dc02b2bc5ae5988e94cbe6d7ac23cea |
| src/set_attention_model.py | library and pilot source | 59328 | d7a7e72a2a111bb40e45cbc61ed844349e51ed15c7c22749a817893eb8eee345 |
| notebooks/V3 Benchmarks 2080.ipynb | benchmark notebook, 2080 | 297246 | 89c86833deae2dd165dd75e7f4dc4997398f60f042da1787e6ff7657878fb34d |
| notebooks/V3 Benchmarks H100.ipynb | benchmark notebook, H100 | 280375 | b9426a038ede7c4c3a3cc06b4b30a1baa38ff3fd26d7019347a898d95e0493c3 |
| notebooks/V3 Pilot 2080.ipynb | pilot notebook | 174694 | 3451a59fa0717f376da69c5b761710e1f3636bd1b4f5383c8f246f1bd4331643 |
| notebooks/V3_Benchmarks_Tables.ipynb | tables, paragraph and figure notebook | 169903 | 3a197d27ccc9a5187f1c1635deb3163809a0e6c236ed87903a57aa145c8f03ff |
| inputs/v2_fixed_size_run_comparison_attention_n300_M64000.pkl.lz4 | benchmark inputs | 5500887 | 2c465e03354342bb41455e6fbcfdda4e3e07c043cf757b7a92b0689829fbef8e |
| inputs/v2_fixed_size_run_comparison_attention_n496_M64000.pkl.lz4 | benchmark inputs | 8538001 | 1f854c8f5390551002dae75167e71f7981e4a95ecd06be90d549e96bcc53db8d |
| inputs/v2_limited_sharing_control_attention_n300_M64000.pkl.lz4 | benchmark inputs | 9343038 | 1d8c93617f2806604694c75d2c1bee022cf6d0586cb919f70a385a39854bb9ea |
| inputs/v2_limited_sharing_control_attention_n496_M64000.pkl.lz4 | benchmark inputs | 16043265 | 85a2bd747a0e034b32643015043e7bc50b27def4d4f39658bfd0bf520f0c66ca |
| inputs/v2_main_comparison_attention_n300_M128000.pkl.lz4 | benchmark inputs | 5970898 | 0fb565b342ccc9c7ffa2ccd1730d85d7b4e3e5f4e95d5f184370ab60cb9e9a6d |
| inputs/v2_main_comparison_attention_n300_M256000.pkl.lz4 | benchmark inputs | 7340157 | 81e482658124baaf9495624b966a774ce423180ffff761425e542cd583e6ffe2 |
| inputs/v2_main_comparison_attention_n300_M512000.pkl.lz4 | benchmark inputs | 10053533 | 5e400c03791467826531ef7de6f2e170b2b69bf3f0491d649fec6d5d7ff04e1e |
| inputs/v2_main_comparison_attention_n300_M64000.pkl.lz4 | benchmark inputs | 5284510 | 31f07a0a50e563ab6aa7ad229385796feaadd6a56c5ebe87e95db684d9cf16af |
| inputs/v2_main_comparison_attention_n496_M128000.pkl.lz4 | benchmark inputs | 9082624 | 79cf75564049c73ee4ac49170c9e2eb43caf468ba9a530526edbf1698623df34 |
| inputs/v2_main_comparison_attention_n496_M256000.pkl.lz4 | benchmark inputs | 10591732 | 87a930997c72062343f893d0ce82e9f37a50334719f1fdd31d2e5099c3e28d73 |
| inputs/v2_main_comparison_attention_n496_M512000.pkl.lz4 | benchmark inputs | 13700614 | 766d7057c868d30028f2eb1cb3fb7ed9d26339ad366f03311bd50564ba954a5f |
| inputs/v2_main_comparison_attention_n496_M64000.pkl.lz4 | benchmark inputs | 8304215 | 95bbb5275cad59f964561947b777065f6a224cf4f8cbd321523a47a64c2a01a4 |
| inputs/v2_main_comparison_step_n300_M512000.pkl.lz4 | benchmark inputs | 93943318 | 8992e63539943cbb60c9149be1ac2aaf5209e3e7d19ce4c63402b2c5920046bb |
| inputs/v2_main_comparison_step_n300_M64000.pkl.lz4 | benchmark inputs | 89202574 | 9a7d82c7847899ea9bb0c19810f98e463f4cf2690f11e614048fa4fe05e719bb |
| inputs/v2_main_comparison_step_n496_M512000.pkl.lz4 | benchmark inputs | 94717789 | 5e6ca5668d901af18b0098c9ab769d2d93e43571b82d451d0fd06a805c02edf3 |
| inputs/v2_main_comparison_step_n496_M64000.pkl.lz4 | benchmark inputs | 89512413 | 7959adb2034a44912ea8abb662aa545e09c907c51dfde9fe2893cc2b264dcb70 |
| inputs/v2_small_record_boundary_attention_n14_M16383.pkl.lz4 | benchmark inputs | 352704 | cb9249fe03bf6f85fd05841fc885959a3f2ef70f15045f4cc2e798ff0ed0d7bc |
| inputs/v2_small_record_boundary_attention_n16_M65535.pkl.lz4 | benchmark inputs | 774085 | 7680106e1ab4bb834380305677f2e74f52c966ff6180ab4c74211dc973c6e983 |
| results/section7_benchmarks_2080.pkl.lz4 | benchmark results, 2080 | 103156 | a10bc233806e0d965aab936255034a7c177bb5644f141c0e4c58a558352a3eaf |
| results/h100_v3/section7_benchmarks_H100.pkl.lz4 | benchmark results, H100 | 120794 | 0081c91c8e619e13a82e62ea6bf7699c24314725ee27fe9e2b77e4bc69eeb133 |
| results/tables/derived.pkl.lz4 | generated benchmark tables, paragraph, figure and derived views | 17336 | 83f5153e25a3f271b9871d4bff87db51a0bfeee58099406017c475a308c124e1 |
| results/tables/derived_best_mode.pkl.lz4 | generated benchmark tables, paragraph, figure and derived views | 6415 | 46af91304c13d08e70747793fec0c1308846170ccbd6526ab00fcaf4ad1d21b0 |
| results/tables/fig1_speedup.png | generated benchmark tables, paragraph, figure and derived views | 89338 | 52f2fb3cf18517f94febed83c3d323ae519a5dc329d358a6d86bbd2afdc4cb39 |
| results/tables/fig2_memory.png | generated benchmark tables, paragraph, figure and derived views | 107713 | 9b54e9c2378b0ed068af57777d79d29c3b6a3a7791ff7d89d295e3a7d5a02a58 |
| results/tables/paragraph.tex | generated benchmark tables, paragraph, figure and derived views | 1670 | 8f978c5b973c189d20536a7f9751f460583465c800a89835f0e6c54cdc2f9971 |
| results/tables/tab_boundary.tex | generated benchmark tables, paragraph, figure and derived views | 1454 | d8641501359dfb07afcd641f4886b7e724b44dcd230c091f935aeefd1a966a5e |
| results/tables/tab_ceiling.tex | generated benchmark tables, paragraph, figure and derived views | 908 | 1dc9c76f8be47042703d137f38f0991891dcef1bbafa5a4545963e87fd5187f7 |
| results/tables/tab_control.tex | generated benchmark tables, paragraph, figure and derived views | 2114 | 88ca92ea3b6d65b9b46bfac813f489184053994e005a314f30bc8520fb5b0ecf |
| results/tables/tab_main_attention.tex | generated benchmark tables, paragraph, figure and derived views | 1956 | a2c40ec4d645d1a1c098d165c343176f902c80a4707d89d2c2738cf2f2f6a8aa |
| results/tables/tab_main_step.tex | generated benchmark tables, paragraph, figure and derived views | 1269 | ac34358b82991dd93f52c78b7aaeb3d24fe5aec0ee7222365b494074ac73a151 |
| results/tables/tab_prep_attention.tex | generated benchmark tables, paragraph, figure and derived views | 2837 | c9e39d059be0a8c64fc68a6ad7f7d1a30d2a11854b4d511f7bd58567dd642620 |
| results/tables/tab_prep_step.tex | generated benchmark tables, paragraph, figure and derived views | 2292 | 5f4ab8f74118f157d6b0b9bb9937a01fddcd857e8e53a9055c4f4ebcdd7d1002 |
| results/tables/tab_replay_setup.tex | generated benchmark tables, paragraph, figure and derived views | 2335 | 5ae0186672022aaca882627ccea91e8b6776af55e70b9b428e063b54ef0fc694 |
| results/tables/tab_totals.tex | generated benchmark tables, paragraph, figure and derived views | 1945 | acfe11e8e74fbaa1dfec36acbcfd4fdbe2708c576bcc10ae4b761ec197c79f60 |
| pilot/mc3_pilot_data_d5.pkl.lz4 | pilot data extract | 96715497 | aa462904622bd71701d796a2f5166db9789c92b31ce40f451b2c641ffe6889a2 |
| pilot/mc3_contexts_d5.pkl.lz4 | pilot data extract | 40567742 | e5b25a32a6f9994623574091a0e41c8645d10201ea143219386b95c752f41373 |
| pilot/mc3_decile_edges.pkl.lz4 | pilot data extract | 614 | f6270354a9b668813e5a19f4dd9f296d3072ac118d4a53dfb4dc443a3d4ed1ec |
| pilot/mc3_pilot_data_d5_removed.pkl.lz4 | pilot data extract | 1053372 | 2281213e46360dd51c4d8a3d04243f8f6be2721c848931485f84b2c78aeff116 |
| pilot/runs/v3_W0_curve.pkl.lz4 | pilot curve | 2177875 | ac3832644be20cfecb4c1bbda2415e1916d2c06478150e0ece7a0bd377454dee |
| pilot/runs/v3_W0_registered_curve.pkl.lz4 | pilot curve | 2174085 | 9123c0400f10760167dd108c42c16ed800a55548ee6825a3c2cf3e4d0c205b67 |
| pilot/runs/v3_W_curve.pkl.lz4 | pilot curve | 2182163 | b8d04248d2a194b5de8b24e01f5b8adc3882df3f9b1ec84951d83a41bd9ff381 |
| pilot/runs/v3_W0_last.pt | pilot final checkpoint | 22896514 | f128c57e1c504a70c9ff4b4fc58b17a35b503b6af85b38af5fb1be650b5a4ea2 |
| pilot/runs/v3_W0_registered_last.pt | pilot final checkpoint | 22913416 | 2668f09389e13757122fb4343c7a19974b9d014b29e738b5e3551bb14e1b3ed1 |
| pilot/runs/v3_W_last.pt | pilot final checkpoint | 22881584 | 60be33797f637b78a928badd765f1b67b7ed1b7629678d5b40ec50c858565c6e |
| pilot/runs/v3_pilot_report.pkl.lz4 | pilot report | 3701 | 0a285575ee6a8dfd58dd582f128144e4bf350df224adca230b75d241601deda7 |
| pilot/runs/v3_cost_study.pkl.lz4 | pilot cost study | 3740 | dab57d5e88a2f098de28f765ddfe88b37a53c0a0e9b27953a6c3b0c0ccf19dde |
| pilot/runs/v3_gate.pkl.lz4 | pilot agreement gate record, module hashes before training | 3869 | e2da89f20064f75dab24f0bcff57e717e8a60a15a020ca8c94771a42ede7b32b |
| pilot/tables/tab_pilot_baseline.tex | generated pilot tables | 1026 | a9ce77a4fc89b4dab7d054dad6484d11288d35f49a2ff9d20a0683985b15e9aa |
| pilot/tables/tab_pilot_bins_sampled.tex | generated pilot tables | 1764 | 56aba018f882de1cb48d9d7a2265d3226eb6a98abf63a92a96f874e0dc1b1d09 |
| pilot/tables/tab_pilot_endpoints.tex | generated pilot tables | 830 | 437d31dc8cae7437a58309f99bad7f94e0292f77e5bdfae8339401b6f39f66e3 |

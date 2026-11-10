# Release manifest

Generated 2026-11-10 12:57. Paths are relative to the release root. SHA256 is the full digest.

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
| README.md | release README | 19542 | d0ebbafc335458a40700f6bf2725c6259014d7b8929f18b9f2eae8db02b0f195 |
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
| notebooks/V3 Benchmarks 2080.ipynb | benchmark notebook, 2080 | 284577 | e76ac13576415c37c49986d32c2d5e100dbcfcf59b63581720b2a7ebfc387247 |
| notebooks/V3 Benchmarks H100.ipynb | benchmark notebook, H100 | 155681 | ead5e2ad7cc12eb72e24eac14d20a498e61ef6d552f629c7500301ef52b15f3e |
| notebooks/V3 Pilot 2080.ipynb | pilot notebook | 184072 | a62b0f4dcdcd2d1a3303f76a365c9fa583ae928b3b3d671b0313b8f5086c198e |
| notebooks/V3_Benchmarks_Tables.ipynb | tables, paragraph and figure notebook | 157788 | c5975de91e1cc3aab48a337968ca6de2c117aa4ba46b75b044816ff57342565e |
| notebooks/V3 Release Manifest.ipynb | release manifest notebook | 87143 | 561052bfa17453839ce913f58f83959a07690c937e392340d506b07328543ff9 |
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
| results/section7_benchmarks_2080.pkl.lz4 | benchmark results, 2080 | 85794 | 6463ce426816b440af7fcf28382fdde9e8f5787a2837d1732539027a24b73c9e |
| results/h100_v3/section7_benchmarks_H100.pkl.lz4 | benchmark results, H100 | 97536 | 16b5db6768b7891028d827ae19d3772f8f61fd4b1c25f8fecefd0cc6bfe920ed |
| results/tables/derived.pkl.lz4 | generated benchmark tables, paragraph, figure and derived views | 16390 | 7beece8f566d7a86f98d7846aef1e2dfe96c1fa7d80b92272cde94818b8dd1c2 |
| results/tables/derived_best_mode.pkl.lz4 | generated benchmark tables, paragraph, figure and derived views | 6445 | bb883c40f3da518f96a3d8b40cabfbf086b6f5a32d14294f33372dda9f4358d1 |
| results/tables/fig1_speedup.png | generated benchmark tables, paragraph, figure and derived views | 145463 | b0be031138454cd1d82341e066a8452c5f2ab2e9cdaf45e54ccd8073f59e144f |
| results/tables/paragraph.tex | generated benchmark tables, paragraph, figure and derived views | 1670 | d0f60c7b4460c6abc2cb95e74614a3d60d3ca8a28fc76bc01b9b99dbb22ea566 |
| results/tables/tab_boundary.tex | generated benchmark tables, paragraph, figure and derived views | 2153 | 350a9bac09089001b70e60368dc3092c6a23700a848b9f3e78d6ff8335736190 |
| results/tables/tab_ceiling.tex | generated benchmark tables, paragraph, figure and derived views | 906 | 8af131f2537c79bb6866e4c47a66a958fb363dad94127121a527f0ade1f318c0 |
| results/tables/tab_control.tex | generated benchmark tables, paragraph, figure and derived views | 2260 | bc2e0e909499165ae19e880513ff81230f7850039fcd88693afe99ec10cb216f |
| results/tables/tab_main_attention.tex | generated benchmark tables, paragraph, figure and derived views | 3221 | b6e50127603382b7b053dc3cc732f005f7ba9af89acf262bd02875ff4b68d3b4 |
| results/tables/tab_main_step.tex | generated benchmark tables, paragraph, figure and derived views | 2485 | 0b4c276bddcdf7e9e609d45afad4d7896fac6d03dbc3f41be9cbc16792d52e59 |
| results/tables/tab_prep_attention.tex | generated benchmark tables, paragraph, figure and derived views | 2887 | c429e3760ff27dffbd97b7100cd7033e41ec8214afe2bdac8ce76ee5493f4b91 |
| results/tables/tab_prep_step.tex | generated benchmark tables, paragraph, figure and derived views | 2527 | 44d88f7da84dffb15ec7da1282dd7ebce01f42bb93774b78747cf8434691928d |
| results/tables/tab_replay_setup.tex | generated benchmark tables, paragraph, figure and derived views | 2333 | b608f67e94d0297848f5c0808e55099e40b518149c4fb4c228072b4b07904718 |
| results/tables/tab_totals.tex | generated benchmark tables, paragraph, figure and derived views | 2520 | fc1b8101d586b888616cfd8d232dd11c22e066a490f1b6e347b835469e0ef328 |
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
| pilot/tables/tab_pilot_baseline.tex | generated pilot tables | 1022 | 94c31e875669c53d77f3c9fe1d16c28356b0a8933fa6a625ea0a999e9a1adceb |
| pilot/tables/tab_pilot_bins_sampled.tex | generated pilot tables | 1760 | c3112c18b7c6f16d79c93989b305550732013eaef48e2290e6837df1f255e79a |
| pilot/tables/tab_pilot_endpoints.tex | generated pilot tables | 959 | 283e21c2169950f664bff94f15d93479e3075771ae76446a778aef8273baedad |

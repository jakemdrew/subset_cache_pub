# Pilot data fingerprints (files stay under E:\apriori-ml; not in this repo)

| file | fingerprint | contents |
|---|---|---|
| mc3_pilot_data_d5.pkl.lz4 | facts 1d6ee181e9fd1807 | cleaned pilot data (D5): 10,000 training filings 2019 to 2024 Q2, 1,000 held out filings 2024 Q3 and Q4; 1,234,902 training occurrences; 136,446 held out targets |
| mc3_contexts_d5.pkl.lz4 | bb82e1d833c2983b | top 12 and 16 ranked context facts per fact, fitted on cleaned training filings |
| mc3_pilot_data_d5_removed.pkl.lz4 | 57,069 rows | every occurrence removed by the production artifact rule |
| mc2_pilot_data.pkl.lz4 | facts a62fb1b9b0a3644c | uncleaned pilot data (superseded by D5) |
| GaapFeaturesML name store | 56f270b8e5f2c1b1 | concept name vectors |

Module hashes (inspect based, used in every contract): set_attention_cache f53957e0033d0c72, set_attention_model 647b44ba68fe9dd4, masked_concept_model e8c9321790cbddd3, mc_pilot_data f7b8e10a079077d5, mc_batched 0be0f3440e96db76.

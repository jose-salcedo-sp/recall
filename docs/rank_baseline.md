# Rank baseline (Phase 1)

Frozen 2026-09-21 against the seed corpus (`scripts/seed.py`), concat-merge
retrieve (`retrieve.rs` before RRF), `retrieve_k=32`. Embeddings:
`openai/text-embedding-3-small` via the configured gateway.

Judge later retrieve/filter changes with `scripts/rank_probe.py` on
**answer-chunk rank**, not the noul histogram.

```
scripts/rank_probe.py           # retrieve ranks (this table)
scripts/rank_probe.py --rrf     # after Phase 2 fusion
scripts/rank_probe.py --admit   # noul rank + empty-admit (slow: Laya)
```

## Finding 4 (original four)

| Question | Answer chunk | retrieve@32 | noul rank | empty |
| --- | --- | --- | --- | --- |
| When is my sister Ana's birthday? | Ana's birthday is March 14 | **1** | (see admit run) | — |
| What am I allergic to? | allergic to penicillin | **1** | | — |
| What is the wifi password at the cabin? | cabin is redmaple | **1** | | — |
| What is the atomic mass of ruthenium? | — | n/a | | must be empty |

## Full frozen set (24)

`answer-chunk in retrieve@32: 18/23` answerable. Misses are **all granted**
rows: concat+truncate fills k from personal hybrid and drops mounted hits.

| Question | expect substring | retrieve@32 |
| --- | --- | --- |
| When is my sister Ana's birthday? | Ana's birthday is March 14 | 1 |
| What am I allergic to? | allergic to penicillin | 1 |
| What is the wifi password at the cabin? | wifi password at the cabin is redmaple | 1 |
| What is the atomic mass of ruthenium? | (none) | — |
| When is Luis's birthday? | Luis's birthday is July 2 | 1 |
| Where does Ana live? | Ana lives in Denver | 1 |
| What is my mom's name? | Carmen Salcedo | 1 |
| What is the home wifi password? | Winterthorn-5G with password winterthorn | 1 |
| How old is Maple? | Maple is a six-year-old | 1 |
| Who is my manager? | manager is Elena Voss | 1 |
| When did I start this job? | started this job on 2022-09-12 | 1 |
| What is the cabin lockbox code? | cabin lockbox code is 4218 | 1 |
| When is Maya's birthday? | Maya's birthday is December 5 | 1 |
| What is Maya's work badge PIN? | badge PIN is 3301 | **miss** (granted) |
| What is Maya allergic to? | allergic to shellfish | **miss** (granted) |
| Where is Priya's spare key? | under the blue planter | **miss** (granted) |
| When is Luis's anniversary? | anniversary is June 8 | **miss** (granted) |
| What is my car's license plate? | plate KRN-441 | 1 |
| Who is my dentist? | dentist is Dr. Okonkwo | 1 |
| Where did we stay in Lisbon? | Hotel do Chiado | 1 |
| What is my employee ID? | Employee ID is R-10428 | **miss** (granted) |
| When is the Austin wedding? | wedding in Austin on 2026-10-10 | 1 |
| What is my blood type? | blood type is O positive | 1 |
| How do I take my coffee? | coffee with oat milk | 2 |

Noul ranks for Finding 4 were not captured in this session (`GET /v1/asks/{id}`
500 on the then-running binary). Re-run `scripts/rank_probe.py --admit` after
the filter-pack binary is up.

Phase 2 fusion SQL (`scripts/test_fusion.sql`) passed: granted lexical hit
`quilted-zirc-9` is returned by `search_mounted_for_brain`.

After hybrid granted + RRF @64 (`scripts/rank_probe.py --rrf`): **23/23**
answerable chunks in retrieve@64. Granted misses from concat are now rank 1–2.
Cabin wifi is retrieve@2; home wifi is retrieve@1 (both present).

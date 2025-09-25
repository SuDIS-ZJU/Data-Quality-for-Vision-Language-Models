# Availability-Oriented Papers (Data Quality for LVLMs)

This table lists works primarily addressing data **Availability** issues: scarcity, overload (low-signal bloat), domain imbalance, format / integrity errors, label absence, and redundancy / duplication.

<!--
Columns:
- No. : incremental index
- Reference : paper title (hyperlinked) + optional code link
- Pub.Year : publication year
- Publication : venue (conf./journal/arXiv)
- Data Issues : specific Availability sub-issues
Add new rows at the end; keep numbering sequential. You can later re-order if needed.
-->

| No. | Reference | Pub.Year | Publication | Data Issues |
|-----|-----------|----------|-------------|-------------|
| 1 | [DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108) | 2023 | arXiv | Overload, Redundancy, Quality Filtering |
| 2 | [DoReMi: Optimizing data mixture for language model pretraining](https://arxiv.org/abs/2305.10429) | 2023 | ICML | Mixture Optimization, Imbalance |
| 3 | [LAION-5B: Large-scale open dataset for CLIP training](https://arxiv.org/abs/2210.08402) | 2022 | NeurIPS Datasets | Scale, Coverage, Scarcity Mitigation |
| 4 | [CC12M: Conceptual 12M](https://arxiv.org/abs/2102.08981) | 2021 | arXiv | Scale, Coverage |
| 5 | [The Pile](https://arxiv.org/abs/2101.00027) | 2021 | arXiv | Mixture Curation, Redundancy Control |
| 6 | [CC3M: Conceptual Captions](https://aclanthology.org/P18-1238/) | 2018 | ACL | Coverage, Weak Supervision (Label Missing) |
| 7 | [Automatic data acquisition for deep learning](https://dl.acm.org/doi/10.14778/3476311.3476333) | 2021 | VLDB | Scarcity, Sampling Strategy |
| 8 | [Curriculum learning for large-scale data (Placeholder)](https://example.com) | YYYY | Venue | Scarcity, Imbalance (placeholder) |
| 9 | [Efficient dataset deduplication via hashing (Placeholder)](https://example.com) | YYYY | Venue | Redundancy |
|10 | [Format validation & corruption detection (Placeholder)](https://example.com) | YYYY | Venue | Format Error Detection |

<!--
Issue Tag Guidance:
- Scarcity: insufficient volume / domain under-coverage
- Overload: very large raw pool with low average signal quality
- Imbalance: skewed domain / task / distribution
- Redundancy: exact or near-duplicate samples
- Label Missing: absent / partial annotations in multimodal pairs
- Format Error: corrupted files, invalid metadata, parsing failure
-->

<!--
Contribution Checklist:
[ ] Add paper row
[ ] Use canonical title capitalization
[ ] Provide official venue (conf. acronym / journal / arXiv)
[ ] Use concise comma-separated issue tags
[ ] (Optional) Add code link: Title [[code]](URL)
-->

<!-- Example row template (copy & edit, keep pipes):
| XX | Full Title [[code]](repo_link) | YEAR | VENUE | Scarcity, Imbalance |
-->


# Pew American Trends Panel microdata (not redistributed)

The respondent-level microdata used in this study cannot be redistributed
under Pew Research Center's terms of use. We accessed the processed waves via
the OpinionQA release (Santurkar et al., 2023, "Whose Opinions Do Language
Models Reflect?"; https://github.com/tatsu-lab/opinions_qa), which provides
each American Trends Panel wave in exactly the layout this pipeline expects.
The underlying waves are also available directly from Pew Research Center
(https://www.pewresearch.org/american-trends-panel-datasets/), free
registration required.

Waves used: W26, W27, W29, W32, W34, W36, W41, W42, W43, W45, W49, W50, W54, W82, W92

Place each wave in this directory as:

```
human_resp/American_Trends_Panel_W{wave}/responses.csv   # one row per respondent (QKEY),
                                                         # labeled answers + WEIGHT_W{wave} column
human_resp/American_Trends_Panel_W{wave}/info.csv        # question metadata
human_resp/American_Trends_Panel_W{wave}/metadata.csv
```

(If starting from Pew's SPSS files instead, export to CSV with value labels,
e.g. `pyreadstat.read_sav(..., apply_value_formats=True)`.)

The microdata is only required to (a) rebuild the weighted human ground-truth
caches (`verification/rebuild_weighted_caches.py` → `verification/cache/human_W*.pkl`,
already included in this deposit) and (b) re-run the `--recompute` paths of
`paper_figs/compute_*.py` / `fig_alphabeta*.py` / `verify_cases.py`. All of
their outputs are included, so every figure and number in the paper can be
reproduced without this directory.

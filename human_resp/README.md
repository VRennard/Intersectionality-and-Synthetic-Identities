# Pew American Trends Panel microdata (not redistributed)

The respondent-level microdata used in this study cannot be redistributed
under Pew Research Center's terms of use. Download the following American
Trends Panel waves directly from Pew (https://www.pewresearch.org/american-trends-panel-datasets/),
free registration required:

W26, W27, W29, W32, W34, W36, W41, W42, W43, W45, W49, W50, W54, W82, W92

Place each wave in this directory as:

```
human_resp/American_Trends_Panel_W{wave}/responses.csv   # one row per respondent (QKEY),
                                                         # labeled answers + WEIGHT_W{wave} column
human_resp/American_Trends_Panel_W{wave}/info.csv        # question metadata
human_resp/American_Trends_Panel_W{wave}/metadata.csv
```

`responses.csv` is the wave's SPSS file exported to CSV with value labels
(pandas: `pyreadstat.read_sav(..., apply_value_formats=True)`).

The microdata is only required to (a) rebuild the weighted human ground-truth
caches (`verification/rebuild_weighted_caches.py` → `verification/cache/human_W*.pkl`,
already included in this deposit) and (b) re-run the `--recompute` paths of
`paper_figs/compute_*.py` / `fig_alphabeta*.py` / `verify_cases.py`. All of
their outputs are included, so every figure and number in the paper can be
reproduced without this directory.

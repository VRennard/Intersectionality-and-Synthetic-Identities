# Human additivity ceiling under Kish effective n and nested singles

## What this corrects

The paper calibrates human additivity against a ceiling: the share of pair cells an exactly additive population would win in the same contest, measured with the same noise. The published construction draws the synthetic pair at the cell's raw n, independently of the singles. That construction gives a ceiling of 49.3% (n ≥ 100) and 52.5% (n ≥ 200), and puts real respondents at about 90% of it.

Two features of the survey make that reference too lenient:
- **Weighting.** ATP cells are weighted, and the Kish effective sample size is a median 0.53 of the raw n. The reference should be drawn at the effective n.
- **Nesting.** The respondents in a pair cell are also inside both single cells, so sampling noise in the pair is shared with the singles. The reference should keep that overlap.

## Results

Weighted basis, 14 ATP waves (W49 excluded, as in the published calibration), same cells and contest rule as the paper (`human_additivity_calibration.py`). "Attainment" is the human additive share divided by the ceiling.

| Gate (raw n) | Pair cells | Human additive share | Published ceiling (raw n, independent) | Kish + nested ceiling | Attainment, published | Attainment, Kish + nested |
|---|---|---|---|---|---|---|
| n ≥ 100 | 236,752 | 44.4% | 49.3% | 53.6% | 90% | **82.8%** |
| n ≥ 200 | 146,801 | 47.8% | 52.5% | 56.2% | 91% | **84.9%** |

The ceiling is a Monte Carlo draw. A second, independent run of the same construction gave 83.0% and 84.8%, so the attainment figures carry about ±0.1 points of simulation noise.

Each correction applied alone:
- Kish n with independent singles gives a ceiling of 41.3% (n ≥ 100) and 44.1% (n ≥ 200).
- Nested singles at raw n gives 54.7% and 56.8%.

The Kish + nested ceiling is the right reference for the paper's estimand. Under it, real respondents fall about 9 points (17%) short of perfect composition, rather than 5 points (10%).

## Not affected

- The coefficient medians (α 0.95, β 0.95) are unchanged. A perfectly additive population measured through nested singles gives 0.94 to 0.99, so the paper's noise-immune check holds under every construction.

The revision of the manuscript adopts these figures.

## Reproduce

```
python3 verification/human_ceiling_kish_nested.py
```

This needs the ATP microdata in `human_resp/` (see `human_resp/README.md`) and writes `verification/human_ceiling_kish_nested.json`. Random seed 2026. The committed JSON is the run reported above.

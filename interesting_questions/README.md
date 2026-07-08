# Interesting Questions — Distribution Analysis
*Generated from 15 Pew American Trends Panel waves (W26–W92, 2018–2022)*
*1,506 questions scanned, 208 flagged as bimodal or evenly distributed*

---

## Folder Structure

| Folder | Count | What it captures |
|---|---|---|
| `bimodal/split2_50_50/` | 134 | Binary yes/no questions with near-50/50 splits |
| `bimodal/polar3_outer_peaks/` | 36 | 3-option questions where both outer options dominate, middle is near-empty |
| `bimodal/ushape_extremes/` | 7 unique (3 files — 4 are IDIMPORT_W43 variants with same ID) | 4/5-option Likert with U-shape: strongly agree + strongly disagree both high |
| `even/` | 35 | 3+ option questions where all options are roughly equal (CV < 20%) |

---

## Detection Criteria

### Bimodal — 50/50 split (`split2`)
- Exactly 2 options (after dropping Refused/DK)
- min / max ratio ≥ 0.65 (roughly 39/61 or closer to equal)

### Bimodal — outer peaks (`polar3`)
- Exactly 3 options
- Both outer options (positions 0 and 2) each ≥ 25%
- Balance ratio ≥ 0.55
- Middle option (position 1) < 20%

### Bimodal — U-shape (`ushape`)
- 4 or 5 options
- First and last option each ≥ 25%
- Balance ratio ≥ 0.55
- All middle options each < 20%

### Evenly distributed
- 3+ options
- Coefficient of variation (std/mean of percentages) < 20%

---

## Top Findings

### Most polarized binary questions (split2, closest to 50/50)

| Wave | Question | Split |
|---|---|---|
| W92 | Technology companies: positive or negative effect? | 50.2% / 49.8% |
| W92 | Foreign policy: work with dictators or not? | 49.7% / 50.3% |
| W32 | Do you currently have enough income to lead the life you want? | 50.3% / 49.7% |
| W36 | Will gender parity in top executive roles eventually happen? | 50.4% / 49.6% |
| W43 | School integration vs. local community schools? | 49.6% / 50.4% |
| W42 | Is what you know about medical doctors from school? | 50.2% / 49.8% |
| W34 | Are food additives a serious health risk? | 50.7% / 49.3% |
| W32 | Government should do more vs. leave to individuals? | 51.0% / 49.0% |
| W32 | Can most people be trusted? | 48.8% / 51.2% |
| W34 | Do you favor or oppose animal use in scientific research? | 51.2% / 48.8% |
| W92 | Are banks having a positive or negative effect? | 51.2% / 48.8% |
| W92 | Smaller government / fewer services vs. bigger / more services? | 51.5% / 48.5% |
| W27 | Should businesses be able to replace workers with robots freely? | 51.6% / 48.4% |

### Best polar3 questions (outer peaks, middle near-empty)

| Wave | Question | Distribution |
|---|---|---|
| W43 | Black people treated less fairly than white people? | 52.9% less fairly / 2.0% white less fairly / 45.1% equal |
| W34 | Are organic fruits/vegetables healthier? | 46.3% better / 2.2% worse / 51.4% no difference |
| W36 | Does physical attractiveness help a man's career in politics? | 57.2% helps / 1.8% hurts / 41.1% no difference |
| W36 | Would more women in Congress lead to better outcomes? | 44.7% better / 5.3% worse / 50.0% no difference |
| W36 | Are men or women better at top executive business roles? | 46.5% men / 6.2% women / 47.3% no difference |
| W36 | Are men or women better at running professional sports teams? | 42.8% men / 3.0% women / 54.3% no difference |

### Best U-shape questions (extremes dominate on ordered scale)

| Wave | Question | Distribution |
|---|---|---|
| W26 | Gun accessibility at home — always loaded and accessible? | 34.5% always / 16.7% most of the time / 12.2% sometimes / 36.7% never |
| W43 | How important is your racial background to your self-concept? | 25.1% extremely / 17.8% very / 17.5% moderately / 12.9% a little / 26.7% not at all |
| W41 | Is more interracial marriage a good or bad thing? | 29.4% very good / 19.5% somewhat good / 7.7% somewhat bad / 3.6% very bad / 39.8% neither |

*Note on GUNACCESS_W26: gun owners split sharply — either always keep a loaded accessible gun or never do. Almost nobody is in the middle.*

*Note on IDIMPORT_W43: racial identity importance follows a U-shape — people either anchor strongly on racial identity or feel it is irrelevant to their self-concept, with few in the middle.*

### Best evenly distributed questions (perfect 3-way splits)

| Wave | Question | Distribution |
|---|---|---|
| W45 | News shared by friends/family — one-sided, multi-sided, or none? | 33.0% / 33.0% / 34.0% |
| W27 | Should private citizens be allowed to pilot drones? | 33.2% yes / 32.7% no / 34.1% depends |
| W82 | Will international climate action benefit or harm the US economy? | 34.5% benefit / 33.4% harm / 32.1% no impact |
| W26 | Gun ownership reason: major, minor, or not a reason? (sport/recreation) | 33.5% major / 34.6% minor / 31.9% not a reason |
| W36 | Is being compassionate a help or hindrance for women in business? | 34.7% helps / 31.4% hurts / 33.9% no difference |
| W32 | Is public transportation access a problem in your community? | 31.1% major / 37.1% minor / 31.7% not a problem |
| W92 | Should the government provide more, less, or right amount of assistance to people in need? | 35.3% more / 35.8% less / 28.9% right amount |
| W45 | How often do you get news from radio? | 23.9% often / 28.3% sometimes / 22.7% hardly ever / 25.2% never |

---

## Notes for Paper Use

- **split2 questions** are the richest source of genuine polarization. Many concern technology (robots, drones, AI), governance (government size, social trust), and social policy (school integration, foreign policy). These are questions where America is essentially 50/50.

- **polar3 questions** reveal asymmetric polarization: nearly everyone either takes a strong position OR says "no difference" — the extreme opposite view is nearly empty. This is especially visible on racial fairness and gender-in-leadership questions.

- **U-shape questions** are rare (7 total across 15 waves) but substantively striking: gun access and racial identity both show that people cluster at the extremes of an ordered scale.

- **Even questions** are notable because they reveal genuinely unresolved issues where the public has no consensus view — climate economics, drone regulation, media trust.

- **LLM implication**: evenly distributed and U-shaped questions are likely the hardest for LLMs to simulate accurately. A model with strong demographic priors will over-steer away from these distributions, producing a false consensus where none exists.

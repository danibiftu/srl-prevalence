# Cut-off-free prevalence mapping with a Bayesian location–scale spatial rank likelihood

Code for: Bekalo, D.B.; Esenogho, E. Cut-Off-Free Prevalence Mapping with a Bayesian Location–Scale Spatial Rank Likelihood. *Stats* (under review).

All code is in `srl_prevalence.py`: the method (sampler, cut-off-free prevalence with the clustered Bayesian bootstrap, comparator models, diagnostics), the simulation study (Section 5) and the application to women's body mass index in Ethiopia (Section 6).

## Installation

```
pip install -r requirements.txt
```

## Usage

Run from the folder that contains `data/`:

| Command | Output |
|---|---|
| `python srl_prevalence.py example` | Small demonstration on simulated data |
| `python srl_prevalence.py simulate --reps 100` | Simulation study (resumable) |
| `python srl_prevalence.py summarise` | Simulation summary |
| `python srl_prevalence.py sim-outputs` | Figure 1 and Tables 2–4, A2 |
| `python srl_prevalence.py sim-numbers` | Error ratios and coverage figures in Section 5.3 |
| `python srl_prevalence.py extract` | Application analysis file |
| `python srl_prevalence.py fit --variant hetero` | Location–scale fit (`homo`: location-only) |
| `python srl_prevalence.py prevalence --variant hetero` | Zone prevalence, definitions A and B |
| `python srl_prevalence.py maps` | Figures 2 and 3 |
| `python srl_prevalence.py binomial` | Binomial BYM2 comparison |
| `python srl_prevalence.py compare-binomial` | Zone-level comparison with the binomial model |
| `python srl_prevalence.py calibration` | Table 5 |
| `python srl_prevalence.py zone-direct` | Zone-level direct estimates and standard errors |
| `python srl_prevalence.py scale` | Scale coefficients |
| `python srl_prevalence.py effects` | Covariate effects (Table 6) |
| `python srl_prevalence.py diagnostics` | Figures A1 and A2 |
| `python srl_prevalence.py ppc` | Figure A3 |
| `python srl_prevalence.py adaptation` | Acceptance rates during burn-in |

## Data

The simulation needs only the files in `data/`: the adjacency matrices of the 14 regions and 107 zones of Ethiopia and the zone lookup table.

The survey data cannot be redistributed. To reproduce the application, register with the DHS Program (https://dhsprogram.com) and request the 2024/25 Ethiopia DHS individual recode (`ETIR8AFL.dta`) and GPS data (`ETGE8AFL.shp`); place them in `data/dhs/`. Place the OCHA administrative boundaries (`eth_admin_boundaries.gdb`, Humanitarian Data Exchange) in `data/admin/` and the WHO 2007 BMI-for-age reference (`bfawho2007.txt`, from WHO AnthroPlus) in `data/who2007/`.

## Licence

MIT. The administrative boundary data are © OCHA, licensed CC BY-IGO.

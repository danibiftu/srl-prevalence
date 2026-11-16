# Cut-off-free prevalence mapping with a Bayesian location–scale spatial rank likelihood

Code for: Bekalo, D.B.; Esenogho, E. Cut-Off-Free Prevalence Mapping with a Bayesian Location–Scale Spatial Rank Likelihood. *Stats* (under review).

All code is in `srl_prevalence.py`: the method (sampler, cut-off-free prevalence with the clustered Bayesian bootstrap, comparator models, diagnostics), the simulation study (Section 5) and the application to women's body mass index in Ethiopia (Section 6).



## Data

The simulation needs only the files in `data/`: the adjacency matrices of the 14 regions and 107 zones of Ethiopia and the zone lookup table.

The survey data cannot be redistributed. To reproduce the application, register with the DHS Program (https://dhsprogram.com) and request the 2024/25 Ethiopia DHS individual recode (`ETIR8AFL.dta`) and GPS data (`ETGE8AFL.shp`); place them in `data/dhs/`. Place the OCHA administrative boundaries (`eth_admin_boundaries.gdb`, Humanitarian Data Exchange) in `data/admin/` and the WHO 2007 BMI-for-age reference (`bfawho2007.txt`, from WHO AnthroPlus) in `data/who2007/`.



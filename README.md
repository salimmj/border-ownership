# Border-ownership tuning determines the connectivity between V4 and V1 in the macaque visual system

This repository contains the implementation of a computational model that explores the connectivity between area V4 and V1 in the macaque visual system, based on **Border-Ownership (BO) tuning**. The model investigates how BO-tuned neurons in V4 influence figure-background modulation (FBM) in V1 through feedback connections. The simulations and visualizations are inspired by the findings presented in the paper:

**"Border-ownership tuning determines the connectivity between V4 and V1 in the macaque visual system"**
*Danique Jeurissen, Anne F van Ham, Amparo Gilhuis, Paolo Papale, Pieter R Roelfsema, Matthew W Self*
*PMCID: PMC11496508 | PMID: 39438464*

# Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cd border-ownership
uv run marimo run mpl.py
```

# Static plots
![plots](mpl.png)

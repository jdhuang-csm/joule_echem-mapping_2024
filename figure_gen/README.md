# Figure generation
The scripts and notebooks in this folder generate all figures in the main and supplemental texts. 

## Li-ion battery (LIB)
* `LIB_basic.ipynb`: validation of hybrid measurement and DRT-DOP inversion for the LIB
* `gen_lib_fits.py`: script to generate initial DRT-DOP fits of LIB mapping data; run this before `LIB_mapping.ipynb`
* `LIB_mapping.ipynb`: electrochemical mapping analysis of the LIB

## Protonic ceramic electrochemical cell (PCEC)
* `PCEC_basic.ipynb`: validation of hybrid measurement and DRT-DOP inversion for the PCEC
* `gen_pcec_fits.py`: script to generate initial DRT-DOP fits of PCEC mapping data; run this before `PCEC_mapping.ipynb`
* `PCEC_mapping.ipynb`: electrochemical mapping analysis of the PCEC

## Other
* `simulations.ipynb`: notebook to generate simulated supplemental figures
* `DRT-DOP_separation.py`: script illustrating potential pitfalls and solutions for robust DRT-DOP separation
* `DRT_nullspace.ipynb`: notebook exploring how the null space of the DRT and DRT-DOP matrices affects inversion
* `fig_funcs.py`: convenience module for figure formatting

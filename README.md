# polysome-stats
Library of functions for polysome analysis with example jupyter notebooks.

Miniconda environment (env.yaml) can be installed with the command:

`conda env create -f env.yaml`

However, my advise is to first install mamba in your conda base environment because it speeds up environment solving greatly (https://mamba.readthedocs.io/en/latest/installation.html), after installing mamba you can just run:

`mamba env create -f env.yaml`

## Running the jupyter notebooks

In `polysomes` there are three jupyter notebooks (`example.ipynb`, `example_stats_R.ipynb`, `example_plotting.ipynb`) that should be run in **sequential** order:

1) `example.ipynb`: In the first notebook multiple classifications results from RELION are combined in a large pandas dataframe (thanks to the *[starfile](https://github.com/teamtomo/starfile)* package), neigbor density distribution is calculated, and polysome connections are assigned to the ribosomes. All annotations are added to the dataframe which is finally written out as a .csv.
2) `example_stats_R.ipynb`: Here, the .csv is loaded as an R dataframe and all the statistical models are fitted for class associations in polysomes. For this I used the *[mclogit](https://www.elff.eu/software/mclogit/)* package that can fit *multinomial mixed-effects logistic regression*, where the data is treated as counts of events and random effects of repeated collections/experiments can be modelled. From the fitted models probabilities are calculated with their 95% confidence interval (CI) and those are written out.
3) `example_plotting.ipynb`: I don't know how to plot in R, so its back to the comfort of python to make the final plots! In addition to the probabilites plus CIs, frequencies of events per tomograms are also added to show the variation in the data.

## Citing

[Gemmer *et al.*, Visualization of translation and protein biogenesis at the ER membrane, Nature, 2023](https://www.nature.com/articles/s41586-022-05638-5)

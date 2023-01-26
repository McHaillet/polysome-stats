# polysome-stats
Library of functions for polysome analysis with example jupyter notebooks.

Miniconda environment (env.yaml) can be installed with the command:

`conda env create -f env.yaml`

However, my advise is to first install mamba in your conda base environment because it speeds up environment solving greatly (https://mamba.readthedocs.io/en/latest/installation.html), after installing mamba you can just run:

`mamba env create -f env.yaml`

## Running the jupyter notebooks

In `polysomes` there are three jupyter notebooks (`example.ipynb`, `example_stats_R.ipynb`, `example_plotting.ipynb`) that should be run in **sequential** order:

1) `example.ipynb`: In the first notebook we combine multiple classifications results from RELION into a large pandas dataframe (thanks to the *[starfile](https://github.com/teamtomo/starfile)* package), calculate the neigbor density distribution, and assign polysome connections to all the ribosomes. This is all stored in a large pandas dataframe that is finally stored as a .csv file.
2) `example_stats_R.ipynb`: `example_stats_R.ipynb`: Here, we load the .csv dataframe in R and fit all the statistical models for class associations in polysomes. For this we used the *[mclogit](https://www.elff.eu/software/mclogit/)* package that can fit *multinomial mixed-effects logistic regression*, where the data is treated as counts of events and random effects of repeated collections/experiments can be modelled. From the fitted models we calculate probabilities of events with their 95% confidence interval (CI) and store those. 
3) `example_plotting.ipynb`: I don't know how to plot in R, so we go back to the comfort python to make the final plots! In addition to the probabilites plus CIs we also calculated frequencies of events per tomograms to show the variation in the data.

## Citing

[Gemmer *et al.*, Visualization of translation and protein biogenesis at the ER membrane, Nature, 2023](https://www.nature.com/articles/s41586-022-05638-5)

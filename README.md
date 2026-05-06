# SOCS
This is the codebase for Spatiotemporal Optimal Transport with Contiguous Structures (SOCS), as described in our manuscript *Accurate trajectory inference in time-series spatial transcriptomics with structurally-constrained optimal transport*. 

## Installation
To install SOCS, download this repository directly, and add the folder named `socs` to your python path. The file `socs_env.yaml` can be used to create a conda environment with all the dependencies required to run SOCS, by running:\
`conda env create -f socs_env.yaml`.

## Getting Started
To confirm that SOCS has been installed correctly, run the notebook `simple_test.ipynb`, which runs SOCS on a very small sample of MERFISH data.

## Reproducing Figures
To reproduce Fig. 1 of our manuscript, run the notebooks `lung_processing_notebook.ipynb` and `lung_analysis_notebook.ipynb`. To reproduce Fig. 2 of our manuscript, run the notebook `ovary_analysis_notebook.ipynb`.

## Using SOCS
SOCS is fairly straightforward to use. To use SOCS to do trajectory inference on your spatial transcriptomics data, first format your data as AnnData objects, e.g. `adata`, with gene expression stored as a count table in `adata.X`, spatial x-y coordinates stored in `adata.obsm['spatial']`, and spatially contiguous structure labels stored in `adata.obs['struct']`, and time-point labels stored in `adata.obs['time']`. Perform dimensionality reduction if desired.\
Choose the following parameters: `alpha` (a value between 0 and 1, default 0.5), which trades off spatial consistency with genetic consistency, `eps` (default 0.01), which controls the entropy of the inferred transport map (as `eps` increases, the map is more "spread out"), and `rho` which controls the "unbalancedness" of the problem (default 100).\

In selecting a value of `alpha`, we suggest choosing a value that results in a high proportion of t_1 cells mapping to cells of the same cell type at t_2, and a high value of geometric consistency. Typically, this value is between `alpha=0.1` and `alpha=0.5`, in the CPU-processed case. We find that, when using GPU acceleration, lower values of `alpha` tend to be optimal, between 1E-05 and 1E-03. In selecting a value of `eps`, we pick a value that results in the median degree-1 Hill number of the rows of T is between 1 and 2. We find that this is typically between `eps=0.001` and `eps=0.01`. In selecting a value of `rho`, we find that values around `rho=100` are typically optimal. 

To set up the SOCS problem, initialize the class socs.ot.SOCSModel:\
`socs_model = socs.ot.SOCSModel(adata,'time',expr_key='X_pca',struct_key='struct')`

If any structure labels should not be incorporated into the optimization problem, *e.g.* if label 0 indicates that a cell does not belong to a structure, indicate this by adding the argument `struct_excl=[0]` to the initialization.

If running SOCS on a GPU, significant speedup can be achieved by adding the argument `method='fugw_gpu'` to the initialization.

To run trajectory inference with SOCS, run the command:\
`T = socs_model.infer_map(t1,t2,alpha=alpha,eps=eps,rho1=rho1,rho2=rho2)`,
where `t1` and `t2` are the time point labels stored in `adata.obs['time']`.
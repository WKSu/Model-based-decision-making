# Green Rivers Model-based Decision-making

## Preamble

![](https://i.imgur.com/gdrGisD.png)

[Green Rivers](https://sites.google.com/view/green-rivers/) is an environmental organization advocating for a healthy, sustainable future for people, rivers, wildlife, and the economy in The Netherlands. We stand for the biodiversity and nature recovery in the areas surrounding the IJssel river.

This project is part of the EPA1361 Model-based Decision-Making course taught by Drs. Jan Kwakkel and Hans de Bruijn, and Simon Vydra and Alessio Ciullo.

### Source

This repo (or, rather, the folder `model`), is soft-forked from [quaquel/epa1361_open](https://github.com/quaquel/epa1361_open/tree/master/final%20assignment).

### Authors

- Brennen Bouwmeester
- Batoul Mesdaghi
- Omar Quispel
- Kevin W. Su
- Jason R. Wang

## How-To

See the Directory section below for a summarized description of our directory. Otherwise, the most important analysis files are:

1. Open exploration of the base case (do nothing) with 1000 scenarios to see what happens when nothing is done
2. Open exploration of a sample run with 400 scenarios and 75 policies to understand the output space of the model
3. Multi-Objective Robust Optimization (MORO) on the model
4. Open exploration on the created policies from MORO and selection of feasible policies

These steps correspond with the `*.ipynb` files numerically. Our final report is also included, which includes a more detailed interpretation of our results – including the purpose of this project, and our reasoning, methodology, conclusions, and reflection.

### Directory

The files below are the important ones for running this analysis and a short description of their respective purpose.

```
-\
  data\
  model\
  ------ *Forked files from quaqel/epa1361\_open/final_assignment*
  Outcomes\
  --------- *Saved modelling outputs*
  1-Open-Exploration-Base-Case.ipynb
  2-Open-Exploration-400scenarios-and-75policies.ipynb
  3-Directed-Search-MORO.ipynb
  4-Open-Exploration-MORO-Policies.ipynb
  requirements.txt
  visualization_functions.py
```
Make sure to check `requirements.txt` and ensure your packages are compatible with ours.

Note: Some of the interactive widgets only work in Jupyter Notebook by default. If you want to run them in JupyterLab, you'll need [`jupyterlab-widgets`](https://pypi.org/project/jupyterlab-widgets/).



# coding: utf-8

# In[1]:


# get_ipython().system('jupyter nbconvert --to script visualization_functions.ipynb')


# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# In[ ]:


from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, IntegerParameter, RealParameter)

from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario, SequentialEvaluator)

from ema_workbench.em_framework.evaluators import perform_experiments, optimize
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging, utilities

ema_logging.log_to_stderr(ema_logging.INFO)


# In[ ]:


def histogram_maker(results, outcome, n = 3):
    '''
    This function creates multiple histograms across time and location. 
    
    Parameters
    ----------
    results : dataframe 
    outcome : str
    n : int (time steps)
    '''
    
    locations = ["A.1", "A.2", "A.3", "A.4", "A.5"]
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive']


    print("Plot " + str(outcome) + "(Histogram)")
    for step in range(0, n):
        print("This is timestep " + str(step))
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))

        for i, (ax, place) in enumerate(zip(axes.flatten(), locations)):
            # ax.hist(results[str(place) + "_Expected Annual Damage " + str(step)], color=colors[i])
            ax.hist(results[str(place) + "_" + str(outcome) + " " + str(step)], color=colors[i])
            ax.set_xlim(left = 0)
            ax.set_title(place)

        plt.tight_layout()
        plt.show()


# In[ ]:


# https://stackoverflow.com/a/56253636
def legend_without_duplicate_labels(ax):
    '''
    Helper function to remove duplicate legend labels. 
    
    Parameters
    ----------
    ax = Axes Object
    '''
    
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def mean_outcomes(results, outcomes):
    '''
    This function makes the mean 
    
    Parameters
    ----------
    results : dataframe 
    outcomes : list
    '''
    
    
    
#     # Get the mean for all the results across the scenarios to have a quick look at significant locations
    mean_outcomes_df = results.iloc[:, 52:].apply(np.mean, axis = 0)
    
    locations = ["A.1", "A.2", "A.3", "A.4", "A.5"]
    outcomes = outcomes
    x = [0, 1, 2]
    
    # For the base case it is only necessary to have two plots but if you want to add the costs more plots will be added 
    # max 6 outcomes.
    if len(outcomes) == 2:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,8), sharex=True)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(8,8), sharex=True)
        axes = axes.flatten()
        
    # These criteria are not specific to a location    
    special_criteria = ["Expected Evacuation Costs", "RfR Total Costs"]

    for ax, criteria in zip(axes, outcomes):
        for step in x:
            for place in locations:
                if criteria == "RfR Total Costs":
                    ax.plot(step, mean_outcomes_df[[str(criteria) + " " + str(step)]].values[0], 'ro', c ='y')
                elif criteria == "Expected Evacuation Costs":
                    ax.plot(step, mean_outcomes_df[[str(criteria) + " " + str(step)]].values[0], 'ro', c ='y')
                else:
                    if place == "A.1":
                        ax.plot(step, mean_outcomes_df[[str(place) + "_" + str(criteria) + " " + str(step)]].values[0], 
                                'ro', c="b", label = "A.1")
                    elif place == "A.2":
                        ax.plot(step, mean_outcomes_df[[str(place) + "_" + str(criteria) + " " + str(step)]].values[0], 
                                'ro', c="r", label = "A.2")
                    elif place == "A.3":
                        ax.plot(step, mean_outcomes_df[[str(place) + "_" + str(criteria) + " " + str(step)]].values[0], 
                                'ro', c="g", label = "A.3")
                    elif place == "A.4":
                        ax.plot(step, mean_outcomes_df[[str(place) + "_" + str(criteria) + " " + str(step)]].values[0], 
                                'ro', c="m", label = "A.4")
                    elif place == "A.5":
                        ax.plot(step, mean_outcomes_df[[str(place) + "_" + str(criteria) + " " + str(step)]].values[0], 
                                'ro', c="c", label = "A.5")

        ax.set_xlabel("Time Steps")
        ax.set_ylabel(criteria)
        ax.set_title(str(criteria) + " over the locations", y = 1.1)
        legend_without_duplicate_labels(ax)


    plt.tight_layout()
    plt.show()


# In[ ]:


def aggregate_outcomes(results, outcome):
    '''
    This function creates a new column in the given dataframe with the aggregated scores. It does it inplace. 
    
    Parameters
    ----------
    results : dataframe 
    outcome : str
    
    '''
    
    
    list_outcomes_columns = []
    
    for i in results.columns:
        if outcome in i:
            list_outcomes_columns.append(i)
            
    results["Total " + str(outcome)] = results[list_outcomes_columns].sum(axis = 1)


# In[ ]:


def scatter_maker(results, outcome, n = 3):
    locations = ["A.1", "A.2", "A.3", "A.4", "A.5"]
    
    print("Plot " + str(outcome) + "(Scatterplot)")
    for step in range(0, n):
        print("This is timestep " + str(step))
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))

        for i, (ax, place) in enumerate(zip(axes.flatten(), locations)):
            plt.sca(ax)
            
            if i != 4:
                ax = sns.scatterplot(x="scenario", y=(str(place) + "_" + str(outcome) + " " + str(step)), hue="policy",
                          data=results, legend = False)
            else:
                ax = sns.scatterplot(x="scenario", y=(str(place) + "_" + str(outcome) + " " + str(step)), hue="policy",
                          data=results)
                ax.legend(loc = 'upper right', bbox_to_anchor=(2, 1), fontsize = 8)

            ax.set_xlim(left = 0)
            ax.set_title(place)

        plt.tight_layout()
        plt.show()
#         plt.legend()


# In[ ]:


def pairplot_maker(results, location, n = 3):
    list_loc = []

    for i in results.columns:
        if location in i:
            list_loc.append(i)
            
    # TO-DO:        
    # Add other uncertentainties which do not have A.1 in name, like discount rate and flow.. 
    list_loc.append("policy")
    
    sns.pairplot(results[list_loc], hue='policy',  vars=results[list_loc].iloc[:, 6:-1].keys(), )
    plt.tight_layout()
    plt.show()

def boxplot_histogram_maker(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    outcomes_list = ["Total Expected Number of Deaths", "Total Expected Annual Damage"]
    
    for i, (ax, outcome) in enumerate(zip(axes.flatten(), outcomes_list)):
        ax.boxplot(results[outcome])
        print(str(outcome) + " First quantile: " + str(results[outcome].quantile(q = 0.25)))
        print(str(outcome) + " Mean: " + str(results[outcome].mean()))
        
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    for i, (ax, outcome) in enumerate(zip(axes.flatten(), outcomes_list)):
        ax.hist(results[outcome])
        ax.set_title(outcome)
        
def boxplot_maker(results, outcomes):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    boxplots = {}
    policies = results['policy'].unique()

    for ax, outcome in zip(axes, outcomes):
        for policy in policies:
            values = results[results['policy'] == policy][outcome]
            boxplots[policy] = values

        ax.boxplot([boxplots[policy] for policy in sorted(boxplots.keys())])

        if outcome == 'Total Expected Annual Damage':
            ax.set_yscale('log')
        ax.set_title(outcome)

    plt.show()
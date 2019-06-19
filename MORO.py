# Load dependencies

from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)
from ema_workbench.em_framework import sample_uncertainties
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench import ema_logging
from model.problem_formulation import get_model_for_problem_formulation
from model.dike_model_function import DikeNetwork  # @UnresolvedImport
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, IntegerParameter, RealParameter, Constraint)

from ema_workbench import (
    Model, MultiprocessingEvaluator, Policy, Scenario, SequentialEvaluator)

from ema_workbench.em_framework.evaluators import perform_experiments, optimize
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging, utilities

ema_logging.log_to_stderr(ema_logging.INFO)

# Set up MORO

# Initialize model parameters

dike_model, planning_steps = get_model_for_problem_formulation(5)

# Define robustness functions

# Rebustness score


def robustness(data):
    ''' 
    Returns a robustness score for a value you want to minimize.
    s
    We want a function that returns 0 for the outcome to be in the range that we want and higher otherwise.

    Takes in an array and returns a score for each value of the same array.
    '''

    # Normalize
    mean = np.mean(data)
    # Add a small number so the mean is still considered in the score rather than 0
    iqr = np.quantile(data, 0.75, axis=0) - \
        np.quantile(data, 0.25, axis=0) + mean * 0.005
    score = mean * iqr

    return score


def sumover_robustness(*data):
    '''
    Used to aggregate each outcome's varying location and time into one robustness score.

    Input: multiple n-length (but same) arrays and sums up element-wise into a [1,n] array

    Returns: asks the robustness function to calculate a score for the [1,n] array.
    '''
    return robustness(np.sum(data, axis=1))


# Initialize some vars to make `robustness_functions` a bit more read-able
var_list_damage = ['A.1_Expected Annual Damage 0', 'A.1_Expected Annual Damage 1', 'A.1_Expected Annual Damage 2',
                   'A.2_Expected Annual Damage 0', 'A.2_Expected Annual Damage 1', 'A.2_Expected Annual Damage 2',
                   'A.3_Expected Annual Damage 0', 'A.3_Expected Annual Damage 1', 'A.3_Expected Annual Damage 2',
                   'A.4_Expected Annual Damage 0', 'A.4_Expected Annual Damage 1', 'A.4_Expected Annual Damage 2',
                   'A.5_Expected Annual Damage 0', 'A.5_Expected Annual Damage 1', 'A.5_Expected Annual Damage 2']
var_list_deaths = ['A.1_Expected Number of Deaths 0', 'A.1_Expected Number of Deaths 1', 'A.1_Expected Number of Deaths 2',
                   'A.2_Expected Number of Deaths 0', 'A.2_Expected Number of Deaths 1', 'A.2_Expected Number of Deaths 2',
                   'A.3_Expected Number of Deaths 0', 'A.3_Expected Number of Deaths 1', 'A.3_Expected Number of Deaths 2',
                   'A.4_Expected Number of Deaths 0', 'A.4_Expected Number of Deaths 1', 'A.4_Expected Number of Deaths 2',
                   'A.5_Expected Number of Deaths 0', 'A.5_Expected Number of Deaths 1', 'A.5_Expected Number of Deaths 2']
var_list_dike = ['A.1_Dike Investment Costs 0', 'A.1_Dike Investment Costs 1', 'A.1_Dike Investment Costs 2',
                 'A.2_Dike Investment Costs 0', 'A.2_Dike Investment Costs 1', 'A.2_Dike Investment Costs 2',
                 'A.3_Dike Investment Costs 0', 'A.3_Dike Investment Costs 1', 'A.3_Dike Investment Costs 2',
                 'A.4_Dike Investment Costs 0', 'A.4_Dike Investment Costs 1', 'A.4_Dike Investment Costs 2',
                 'A.5_Dike Investment Costs 0', 'A.5_Dike Investment Costs 1', 'A.5_Dike Investment Costs 2']
var_list_rfr = ['RfR Total Costs 0', 'RfR Total Costs 1', 'RfR Total Costs 2']
var_list_evac = ['Expected Evacuation Costs 0',
                 'Expected Evacuation Costs 1', 'Expected Evacuation Costs 2']


def aggregate_outcomes(results, outcome):
    list_outcomes_columns = []

    for i in results.columns:
        if outcome in i:
            list_outcomes_columns.append(i)

    results["Total " + str(outcome)
            ] = results[list_outcomes_columns].sum(axis=1)


# ### Find the ranges for epsilon and hypervolume convergence
#
# To set $\epsilon$ values, we must minimize noise by first running a robust optimize quickly to see a Pareto front develop as stated in section 3.4 of doi: 10.1016/j.envsoft.2011.04.003 (we don't only look at Monte Carlo policies in hope that this will save time).
results = utilities.load_results('Outcomes/400Scenarios75Policies.csv')

experiments, outcomes = results

outcomes = pd.DataFrame(outcomes)
experiments = pd.DataFrame(experiments)
results = experiments.join(outcomes)
results = results.drop(columns="model")

# Aggregate
aggregate_outcomes(outcomes, "Expected Annual Damage")
aggregate_outcomes(outcomes, "Dike Investment Costs")
aggregate_outcomes(outcomes, "Expected Number of Deaths")
aggregate_outcomes(outcomes, "RfR Total Costs")
aggregate_outcomes(outcomes, "Expected Evacuation Costs")

everything = pd.DataFrame(experiments["policy"]).join(outcomes)

# Run robustness, find the 75th quantile (or 0,max)
robust_values = everything.groupby(
    by=["policy"]).apply(robustness).iloc[:, -5:]
hyp_ranges_min = robust_values.apply(np.min)
hyp_ranges_max = robust_values.apply(np.max)
robust_values.quantile(0.75)

# Define model parameters so we can run robust_optimize and find valid epsilon values.
MINIMIZE = ScalarOutcome.MINIMIZE

# These functions need to only return one value...

robustness_functions = [
    ScalarOutcome('Damage Score', variable_name=var_list_damage,
                  function=sumover_robustness, kind=MINIMIZE),
    ScalarOutcome('Deaths Score', variable_name=var_list_deaths,
                  function=sumover_robustness, kind=MINIMIZE),
    ScalarOutcome('Dike Invest Score', variable_name=var_list_dike,
                  function=sumover_robustness, kind=MINIMIZE),
    ScalarOutcome('RfR Invest Score', variable_name=var_list_rfr,
                  function=sumover_robustness, kind=MINIMIZE),
    ScalarOutcome('Evac Score', variable_name=var_list_evac,
                  function=sumover_robustness, kind=MINIMIZE),
]

constraints = [Constraint("discount_for_rfr_0", outcome_names="RfR Total Costs 0",
                          function=lambda x:max(0, x-426.24)),
               Constraint("discount_for_rfr_1", outcome_names="RfR Total Costs 1",
                          function=lambda x:max(0, x-284.16)),
               Constraint("discount_for_rfr_2", outcome_names="RfR Total Costs 2",
                          function=lambda x:max(0, x-142.08))]

# from ema_workbench import ema_logging
# from ema_workbench.em_framework.optimization import (HyperVolume,
#                                                      EpsilonProgress)
# from ema_workbench.em_framework.evaluators import BaseEvaluator

# BaseEvaluator.reporting_frequency = 0.1
# ema_logging.log_to_stderr(ema_logging.INFO)

# epsilons = [0.05,]*len(robustness_functions)

# start = time.time()

# with MultiprocessingEvaluator(dike_model) as evaluator:
#     results, convergence = evaluator.robust_optimize(robustness_functions,
#                                                      scenarios=10,
#                                                      nfe=200,
#                                                      epsilons=epsilons,
#                                                      convergence=[EpsilonProgress()],
#                                                      convergence_freq=1,
#                                                      constraint=constraints
#                                                     )

# end = time.time()
# print("Time taken: {:0.5f} minutes".format((end - start)/60))

# with open('Outcomes/initial_Pareto_policies.pkl', 'wb') as file_pi:
#     pickle.dump(results, file_pi)

with open('Outcomes/initial_Pareto_policies.pkl', 'rb') as file_pi:
    results = pickle.load(file_pi)

# Now that we can some policies somewhere on a Pareto front, we can run them under more scenarios and see the variance of their values across those scenarios.
policies = []
for row in range(results.shape[0]):
    policies.append(
        # Do not include the damage scores
        Policy(name=row, **results.iloc[row, :-5].to_dict())
    )

# with MultiprocessingEvaluator(dike_model) as evaluator:
#     results = evaluator.perform_experiments(scenarios=50,policies=policies)

# with open('Outcomes/epsilon_results.pkl', 'wb') as file_pi:
#     pickle.dump(results, file_pi)

with open('Outcomes/epsilon_results.pkl', 'rb') as file_pi:
    results = pickle.load(file_pi)

experiments, outcomes = results

outcomes = pd.DataFrame(outcomes)
experiments = pd.DataFrame(experiments)
results = experiments.join(outcomes)
results = results.drop(columns="model")

aggregate_outcomes(outcomes, "Expected Annual Damage")
aggregate_outcomes(outcomes, "Dike Investment Costs")
aggregate_outcomes(outcomes, "Expected Number of Deaths")
aggregate_outcomes(outcomes, "RfR Total Costs")
aggregate_outcomes(outcomes, "Expected Evacuation Costs")

everything = pd.DataFrame(experiments["policy"]).join(outcomes)

robust_values = everything.groupby(
    by=["policy"]).apply(robustness).iloc[:, -5:]

# Finally, find the IQR ranges that are our 'noise-adjusted epsilon values'
ranges = robust_values.apply(sp.stats.iqr)

# And now we can run the main computationally expensive MORO!

BaseEvaluator.reporting_frequency = 0.1
ema_logging.log_to_stderr(ema_logging.INFO)


n_scenarios = 1  # 50
scenarios = sample_uncertainties(dike_model, n_scenarios)
nfe = int(1)

# The expected ranges are set to minimize noise as discussed in section 3.4 of doi: 10.1016/j.envsoft.2011.04.003
epsilons = ranges.values
convergence = [HyperVolume([0,0,0,0,0], hyp_ranges_max),
               EpsilonProgress()]

# Time the output
start = time.time()

with MultiprocessingEvaluator(dike_model) as evaluator:
    results, convergence = evaluator.robust_optimize(robustness_functions,
                                                     scenarios=scenarios,
                                                     nfe=nfe,
                                                     epsilons=epsilons,
                                                     convergence=convergence,
                                                     convergence_freq=20,
                                                     logging_freq=10,
                                                     constraint=constraints
                                                     )

end = time.time()
print("Time taken: {:0.5f} minutes".format((end - start)/60))


filename = 'Outcomes/MORO_s' + str(n_scenarios) + '_nfe' + str(nfe) + '.pkl'
with open(filename, 'wb') as file_pi:
    pickle.dump((results, convergence), file_pi)

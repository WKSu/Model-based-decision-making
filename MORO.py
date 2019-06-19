# Load dependencies

from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)
import pickle
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench import ema_logging
from ema_workbench.em_framework import sample_uncertainties
from model.problem_formulation import get_model_for_problem_formulation
from model.dike_model_function import DikeNetwork  # @UnresolvedImport
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
    iqr = sp.stats.iqr(data) + mean * 0.005
    score = mean * iqr

    return score


def sumover_robustness(*data):
    '''
    Used to aggregate multiple outcomes into one robustness score.

    Input: multiple n-length (but same) arrays and sums up element-wise into a [1,n] array

    Returns: asks the robustness function to calculate a score for the [1,n] array.
    '''
    return robustness(sum(data))


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

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE

# These functions need to only return one value...

robustness_functions = [
    ScalarOutcome('Damage Score', variable_name=var_list_damage,
                  function=sumover_robustness, kind=MINIMIZE, expected_range=(0, 4e16)),
    ScalarOutcome('Deaths Score', variable_name=var_list_deaths,
                  function=sumover_robustness, kind=MINIMIZE, expected_range=(0, 8.5e19)),
    ScalarOutcome('Dike Invest Score', function=sumover_robustness,
                  kind=MINIMIZE, variable_name=var_list_dike, expected_range=(1e18, 1.3e7)),
    ScalarOutcome('RfR Invest Score', variable_name=var_list_rfr,
                  function=sumover_robustness, kind=MINIMIZE, expected_range=(2e16, 9.1e17)),
    ScalarOutcome('Evac Score', variable_name=var_list_evac,
                  function=sumover_robustness, kind=MINIMIZE, expected_range=(0, 2.5e12)),
]

constraints = [Constraint("discount_for_rfr_0", outcome_names="RfR Total Costs 0",
                          function=lambda x:max(0, x-426.24)),
               Constraint("discount_for_rfr_1", outcome_names="RfR Total Costs 1",
                          function=lambda x:max(0, x-284.16)),
               Constraint("discount_for_rfr_2", outcome_names="RfR Total Costs 2",
                          function=lambda x:max(0, x-142.08))]


n_scenarios = 50
scenarios = sample_uncertainties(dike_model, n_scenarios)
nfe = int(4000)


BaseEvaluator.reporting_frequency = 0.1
ema_logging.log_to_stderr(ema_logging.INFO)

epsilons = [0.05, ]*len(robustness_functions)
convergence = [HyperVolume(minimum=[0, 0, 0, 0, 0], maximum=[4e20, 8.5e25, 1.3e20, 9.1e20, 2.5e25]),
               EpsilonProgress()]
# .from_outcomes(robustness_functions)
# minimum=[0,0,0,0,0], maximum=[4e20, 8.5e25, 1.3e20, 9.1e20, 2.5e25])
start = time.time()

with MultiprocessingEvaluator(dike_model) as evaluator:
    results, convergence = evaluator.robust_optimize(robustness_functions,
                                                     scenarios=scenarios,
                                                     nfe=nfe,
                                                     epsilons=epsilons,
                                                     convergence=convergence,
                                                     convergence_freq=20,
                                                     logging_freq=1,
                                                     constraint=constraints
                                                     )

end = time.time()
print("Time taken: {:0.5f} minutes".format((end - start)/60))

# Save results
with open('outcomes/MORO_s50_nfe4000.pkl', 'wb') as file_pi:
    pickle.dump((results, convergence), file_pi)

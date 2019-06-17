# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:18:05 2017

@author: ciullo
"""
from __future__ import division
from copy import deepcopy
from ema_workbench import ema_logging

from model import funs_generate_network
from model.funs_dikes import Lookuplin, dikefailure, init_node
from model.funs_economy import cost_fun, discount, cost_evacuation
from model.funs_hydrostat import werklijn_cdf, werklijn_inv
import numpy as np
import pandas as pd


def Muskingum(C1, C2, C3, Qn0_t1, Qn0_t0, Qn1_t0):
    ''' Simulates hydrological routing '''
    Qn1_t1 = C1 * Qn0_t1 + C2 * Qn0_t0 + C3 * Qn1_t0
    return Qn1_t1


class DikeNetwork(object):
    def __init__(self):
        # planning steps
        self.num_planning_steps = 3
        self.num_events = 30
        
        # load network
        G, dike_list, dike_branch, planning_steps = funs_generate_network.get_network(
                                                            self.num_planning_steps)

        # Load hydrological statistics:
        self.A = pd.read_excel('./data/hydrology/werklijn_params.xlsx')

        lowQ, highQ = werklijn_inv([0.992, 0.99992], self.A)
        self.Qpeaks = np.unique(np.asarray(
            [np.random.uniform(lowQ, highQ) / 6 for _ in range(0, self.num_events)]))[::-1]

        # Probabiltiy of exceedence for the discharge @ Lobith (i.e. times 6)
        self.p_exc = 1 - werklijn_cdf(self.Qpeaks * 6, self.A)

        self.G = G
        self.dikelist = dike_list
        self.dike_branch = dike_branch
        self.planning_steps = planning_steps

        # Accounting for the discharge reduction due to upstream dike breaches
        self.sb = True

        # Planning window [y], reasonable for it to be a multiple of num_planning_steps
        self.n = 200
        # Years in planning step:
        self.y_step = self.n//self.num_planning_steps
        # Step of dike increase [m]
        self.dh = 0.5

        # Time step correction: Q is a mean daily value expressed in m3/s
        self.timestepcorr = 24 * 60 * 60
#        ema_logging.info('model initialized')

    # Initialize hydrology at each node:
    def _initialize_hydroloads(self, node, time, Q_0):
        node['cumVol'], node['wl'], node['Qpol'], node['hbas'] = (
            init_node(0, time) for _ in range(4))
        node['Qin'], node['Qout'] = (init_node(Q_0, time) for _ in range(2))
        node['status'] = init_node(False, time)
        node['tbreach'] = np.nan
        return node

    def _initialize_rfr_ooi(self, G, dikenodes, steps):
        for s in steps:
            for n in dikenodes:
                node = G.node[n]
                # Create a copy of the rating curve that will be used in the sim:
                node['rnew'] = deepcopy(node['r'])

                # Initialize outcomes of interest (ooi):
                node['losses {}'.format(s)] = []
                node['deaths {}'.format(s)] = []
                node['evacuation_costs {}'.format(s)] = []

            # Initialize room for the river
            G.node['RfR_projects {}'.format(s)]['cost'] = 0
        return G

    def progressive_height_and_costs(self, G, dikenodes, steps):  
        for dike in dikenodes:             
            node = G.node[dike]
            # Rescale according to step and tranform in meters
            for s in steps:
                node['DikeIncrease {}'.format(s)] *= self.dh
                # 1 Initialize fragility curve
                # 2 Shift it to the degree of dike heigthening:
                # 3 Calculate cumulative raising
                node['fnew {}'.format(s)] = deepcopy(node['f'])
                node['dikeh_cum {}'.format(s)] = 0
                
                for ss in steps[steps <= s]:
                    node['fnew {}'.format(s)][:, 0] += node['DikeIncrease {}'.format(ss)]                 
                    node['dikeh_cum {}'.format(s)] += node['DikeIncrease {}'.format(ss)]
                
                # Calculate dike heigheting costs:
                if node['DikeIncrease {}'.format(s)] == 0:
                    node['dikecosts {}'.format(s)] = 0
                else:
                    node['dikecosts {}'.format(s)] = cost_fun(
                                node['traj_ratio'],
                                node['c'],
                                node['b'],
                                node['lambda'],
                                node['dikeh_cum {}'.format(s)],
                                node['DikeIncrease {}'.format(s)])

    def __call__(self, timestep=1, **kwargs):

        G = self.G
        Qpeaks = self.Qpeaks
        dikelist = self.dikelist

        # Call RfR initialization:
        self._initialize_rfr_ooi(G, dikelist, self.planning_steps)

        # Load all kwargs into network. Kwargs are uncertainties and levers:
        for item in kwargs:
            # when item is 'discount rate':
            if 'discount rate' in item:
                G.node[item]['value'] = kwargs[item]
            # the rest of the times you always get a string like {}_{}:
            else:
                string1, string2 = item.split('_')

                if 'RfR' in string2:
                    # string1: projectID
                    # string2: rfr #step
                    # Note: kwargs[item] in this case can be either 0
                    # (no project) or 1 (yes project)
                    temporal_step = string2.split(' ')[1]
                    
                    proj_node = G.node['RfR_projects {}'.format(temporal_step)]
                    # Cost of RfR project
                    proj_node['cost'] += kwargs[item] * proj_node[string1][
                        'costs_1e6'] * 1e6

                    # Iterate over the location affected by the project
                    for key in proj_node[string1].keys():
                        if key != 'costs_1e6':
                            # Change in rating curve due to the RfR project
                            G.node[key]['rnew'][:, 1] -= kwargs[item] * proj_node[
                                string1][key]
                else:
                    # string1: dikename or EWS
                    # string2: name of uncertainty or lever
                    G.node[string1][string2] = kwargs[item]
                    
        self.progressive_height_and_costs(G, dikelist, self.planning_steps)

        # Percentage of people who can be evacuated for a given warning
        # time:
        G.node['EWS']['evacuation_percentage'] = G.node['EWS']['evacuees'][
            G.node['EWS']['DaysToThreat']]

        # Dictionary storing outputs:
        data = {}
        
        for s in self.planning_steps:
            for Qpeak in Qpeaks:
                node = G.node['A.0']
                waveshape_id = node['ID flood wave shape']

                time = np.arange(0, node['Qevents_shape'].loc[waveshape_id].shape[0],
                             timestep)
                node['Qout'] = Qpeak * node['Qevents_shape'].loc[waveshape_id]

                # Initialize hydrological event:
                for key in dikelist:
                    node = G.node[key]

                    Q_0 = int(G.node['A.0']['Qout'][0])

                    self._initialize_hydroloads(node, time, Q_0)
                    # Calculate critical water level: water above which failure
                    # occurs
                    node['critWL'] = Lookuplin(node['fnew {}'.format(s)], 1, 0, node['pfail'])

                # Run the simulation:
                # Run over the discharge wave:
                for t in range(1, len(time)):
                    # Run over each node of the branch:
                    for n in range(0, len(dikelist)):
                        # Select current node:
                        node = G.node[dikelist[n]]
                        if node['type'] == 'dike':

                            # Muskingum parameters:
                            C1 = node['C1']
                            C2 = node['C2']
                            C3 = node['C3']

                            prec_node = G.node[node['prec_node']]
                            # Evaluate Q coming in a given node at time t:
                            node['Qin'][t] = Muskingum(C1, C2, C3,
                                                   prec_node['Qout'][t],
                                                   prec_node['Qout'][t - 1],
                                                   node['Qin'][t - 1])

                            # Transform Q in water levels:
                            node['wl'][t] = Lookuplin(
                                            node['rnew'], 0, 1, node['Qin'][t])

                            # Evaluate failure and, in case, Q in the floodplain and
                            # Q left in the river:
                            res = dikefailure(self.sb,
                                          node['Qin'][t], node['wl'][t],
                                          node['hbas'][t], node['hground'],
                                          node['status'][t - 1], node['Bmax'],
                                          node['Brate'], time[t],
                                          node['tbreach'], node['critWL'])

                            node['Qout'][t] = res[0]
                            node['Qpol'][t] = res[1]
                            node['status'][t] = res[2]
                            node['tbreach'] = res[3]

                            # Evaluate the volume inside the floodplain as the integral
                            # of Q in time up to time t.
                            node['cumVol'][t] = np.trapz(
                                    node['Qpol']) * self.timestepcorr

                            Area = Lookuplin(node['table'], 4, 0, node['wl'][t])
                            node['hbas'][t] = node['cumVol'][t] / float(Area)

                        elif node['type'] == 'downstream':
                            node['Qin'] = G.node[dikelist[n - 1]]['Qout']

                # Iterate over the network and store outcomes of interest for a
                # given event
                for dike in self.dikelist:
                    node = G.node[dike]

                # If breaches occured:
                    if node['status'][-1] == True:
                        # Losses per event:
                        node['losses {}'.format(s)].append(Lookuplin(node['table'],
                                                    6, 4, np.max(node['wl'])))

                        node['deaths {}'.format(s)].append(Lookuplin(node['table'],
                                                    6, 3, np.max(node['wl'])) * (
                                    1 - G.node['EWS']['evacuation_percentage']))

                        node['evacuation_costs {}'.format(s)].append(
                                cost_evacuation(Lookuplin(
                                node['table'], 6, 5, np.max(node['wl'])
                                ) * G.node['EWS']['evacuation_percentage'],
                                G.node['EWS']['DaysToThreat']))
                    else:
                        node['losses {}'.format(s)].append(0)
                        node['deaths {}'.format(s)].append(0)
                        node['evacuation_costs {}'.format(s)].append(0)

            EECosts = []
            # Iterate over the network,compute and store ooi over all events
            for dike in dikelist:
                node = G.node[dike]

                # Expected Annual Damage:
                EAD = np.trapz(node['losses {}'.format(s)], self.p_exc)
                # Discounted annual risk per dike ring:
                disc_EAD = np.sum(discount(EAD, rate=G.node[
                        'discount rate {}'.format(s)]['value'], n=self.y_step))

                # Expected Annual number of deaths:
                END = np.trapz(node['deaths {}'.format(s)], self.p_exc)

                # Expected Evacuation costs: depend on the event, the higher
                # the event, the more people you have got to evacuate:
                EECosts.append(np.trapz(node['evacuation_costs {}'.format(s)], self.p_exc))

                data.update({'{}_Expected Annual Damage {}'.format(dike,s): disc_EAD,
                         '{}_Expected Number of Deaths {}'.format(dike,s): END,
                         '{}_Dike Investment Costs {}'.format(dike,s
                                              ): node['dikecosts {}'.format(s)]})

            data.update({'RfR Total Costs {}'.format(s): G.node[
                                'RfR_projects {}'.format(s)]['cost'.format(s)]})
            data.update({'Expected Evacuation Costs {}'.format(s): np.sum(EECosts)})

        return data




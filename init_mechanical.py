'''
Initialization (init) script for Spinal Dorsal Horn Network Model
from Sekiguchi et al. (2021)
'''

from netpyne import sim
from neuron import h
					
simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='cfg_mechanical.py', netParamsDefault='netParams_mechanical.py')

# Create, Simulate, and Analyze the Network:
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)


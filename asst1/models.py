import numpy as np
import plotly as pt
import plotly.graph_objects as go
# import matplotlib as plt
import math

class neuron:
    def __init__(self, time = 100, step = 2):
        #all time units are milliseconds
        self.time = time
        self.step = step
        #so simulate, i value in this arr is considered to be 1 step
        #so like element 0 is t = 1 and element n is t = n*step
        self.time_arr = np.arange(0, time+1, step)
        # membrane potential for a given time: it has 1:1 correspondence with time_arr
        self.vm = np.zeros(int(time/step)+1)
        # tells us when spikes via a 1
        self.spikes = np.zeros(int(time/step)+1)
        # just an array to keep track of our voltages to make it easier to plot
        self.v = np.zeros(int(time/step)+1)

    def plot_graph(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.spikes, mode='lines', name='Spike'))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.vm, mode='lines', name='Membrane Potential'))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.v, mode='lines', name='Input Voltage'))
        fig.update_layout(
            title = "Leaky Integrate-and-Fire",
            xaxis_title="Time (ms)",
            yaxis_title="Voltage (mv)"

        )
        fig.show()
        
class lif:
    # rm = resitence, cm = capcitence, input 1 and 2 are our input voltages
    # vt is our threshold voltage and vr is our reset voltage
    def __init__(self, rm, cm, input1 = .2, input2 = .5, vt=1, vr = 0):
        self.neuron = neuron()
        self.rm = rm
        self.cm = cm
        self.tau = rm*cm
        self.input1 = input1
        self.input2 = input2
        self.vt = vt
        self.vr = vr
    
    def current_voltage(self, input, step):
        #for dv i'm not sure if that's the change in voltage from the last step
        #so (curr_voltage-last_step_voltage) or some other shit like step*curr_voltage

        return self.rm*input - self.tau*input*step

    # flip is the time when we want to up voltage: refers to index in time_arr
    def simulate(self, flip):
        step = self.neuron.step
        for i in range(1, len(self.neuron.vm)):
            if i < flip:
                mp = self.neuron.vm[i-1]+self.current_voltage(self.input1, step)
                self.neuron.v[i] = self.input1
            else:
                mp = self.neuron.vm[i-1]+self.current_voltage(self.input2, step)
                self.neuron.v[i] = self.input2
            if mp >= self.vt:
                self.neuron.spikes[i] = 1
                self.neuron.vm[i] = self.vr
            else:
                 self.neuron.vm[i] = mp


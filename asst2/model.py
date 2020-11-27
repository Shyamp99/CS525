import numpy as np
import plotly as pt
import plotly.graph_objects as go
import matplotlib as plt
import math

class neuron:
    def __init__(self, vt, time = 100, step = .5):
        # all time units are milliseconds
        self.time = time
        self.step = step
        #our spike threshold
        self.vt = vt
        # so simulate, i value in this arr is considered to be 1 step
        # so element 0 is t = 0 and element n is t = n*(1/step)
        self.time_arr = np.arange(0, time+1, step)
        # membrane potential for a given time: it has 1:1 correspondence with time_arr
        self.vm = None
        # tells us when spikes via a 1
        self.spikes = np.zeros(int(time/step)+1)
        # just an array to keep track of our voltages to make it easier to plot
        self.a = np.zeros(int(time/step)+1)
        
        
    # if mode is 0 we are plotting hodkins huxley and don't need a spike threshold line
    def plot_graph(self, title, mode = 1):
        vt_line = self.vt*np.ones(int(self.time/self.step)+1)
        fig = go.Figure()
        if mode:
            fig.add_trace(go.Scatter(x=self.time_arr, y=vt_line, mode='lines', name='Spike Threshold (mV)', line=dict(color="blue", dash='dot')))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.vm, mode='lines', name='Membrane Potential (mV)'))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.a, mode='lines', name='Input Current (nA)'))
        fig.update_layout(
            title = title,
            xaxis_title="Time (ms)",
            yaxis_title="Value"
        )
        fig.show()


class lif:
    # rm = resitance, cm = capcitence,
    # vt is our threshold voltage and vr is our reset voltage
    def __init__(self, rm, cm, inputv = 1, vt=1, vr = 0):
        self.neuron = neuron(vt = vt)
        self.neuron.vm = np.zeros(int(self.neuron.time/self.neuron.step)+1)
        self.rm = rm
        self.cm = cm
        self.tau = rm*cm
        self.inputv = inputv
        self.vt = vt
        self.vr = vr

    # we are just solving for dv here
    def current_voltage(self, input, step, prev):
        return ((-prev + input*self.rm) / self.tau)

    # basically called if teacher nueron or input neuron spikes 
    def check_spike(self, flip, flip_end):
        step = self.neuron.step
        for i in range(1, len(self.neuron.vm)):
            prev = self.neuron.vm[i-1] if self.neuron.vm[i-1]<self.vt else self.vr
            if i < flip:
                mp = prev+self.current_voltage(self.input1, step, prev)
                self.neuron.a[i] = self.input1
            elif (i>=flip and i <= flip_end):
                mp = prev+self.current_voltage(self.input2, step, prev)
                self.neuron.a[i] = self.input2
            elif i > flip_end:
                mp = prev+self.current_voltage(self.vr, step, prev)
                self.neuron.a[i] = self.vr
            if mp >= self.vt:
                self.neuron.spikes[i] = 1
            self.neuron.vm[i] = mp

class AND_model:
    '''
    input_x and y are the 0 or 1 values, zero_fr and one_fr are the firing rates we are encoding 0 and 1 as
    post is our post synaptic neuron
    '''
    def __init__(self, neuron, input_x, input_y, zero_fr = 1, one_fr = 4):
        #initialize both to 0.5 because 1/number of weights since we're using oja's rule
        self.weights = [0.5,0.5]
        self.zero_fr = zero_fr
        self.one_fr = one_fr
        self.post = lif(rm=10, cm=10, inputv = 0, vt=1, vr = 0)
        self.input_x = input_x
        self.input_y = input_y
        
    def train(self, ):
        #using oja's rule here
        # teacher is effectively just going to self.post and yeeting in current
        pass

    def sim(self, time, timestep):
        # rough logic: we loop over time/timestep and each iteration of the loop is a time step
        # it either inducesa spike in x or y which are solely represented via firing rates
        pass

import numpy as np
import plotly as pt
import plotly.graph_objects as go
import matplotlib as plt
import math

class neuron:
    def __init__(self, vt, time, step):
        # all time units are milliseconds
        self.time = time
        self.timestep = step
        #our spike threshold
        self.vt = vt
        # time_arr may be needed for graphing - will remove if not needed
        # self.time_arr = np.arange(0, time+1, step)
        # tells us when spikes via a 1
        self.spikes = np.zeros(int(time/step)+1)
        
        
    # need to implement raster plot via matplotlib or plotly - idt plotly has raster plot
    def plot_graph(self, title):
        pass


class lif:
    # rm = resitance, cm = capcitence,
    # vt is our threshold voltage and vr is our reset voltage
    def __init__(self, rm, cm, time = 5, timestep = 0.25, inputv = 1, vt=1, vr = 0):
        self.neuron = neuron(vt = vt, time = time, step = timestep)
        self.rm = rm
        self.cm = cm
        self.tau = rm*cm
        self.inputv = inputv
        self.vt = vt
        self.vr = vr
        self.mp = 0

    # we are just solving for dv here
    # for updating mp with each iteration
    def update_voltage(self, input, prev):
        return ((-1*self.mp + input*self.rm) / self.tau)

    # basically called if teacher nueron or input neuron spikes 
    def check_spike(self, input_current, time, train = False, label = None):
        if train:
            pass
        else:
            # updating voltage based on formula for LIF/from last assignment
            self.mp += self.update_voltage(input_current, self.mp)
            #if spike we handle reset
            if self.mp >= self.vt:
                self.mp = self.vr
                self.neuron.spikes[time] = 1


class AND_model:
    '''
    input_x and y are the 0 or 1 values, zero_fr and one_fr are the firing rates we are encoding 0 and 1 as
    post is our post synaptic neuron
    '''
    def __init__(self, neuron, input_x, input_y, zero_fr = 1, one_fr = 4, time = 5):
        #initialize both to 0.5 because 1/number of weights since we're using oja's rule
        self.w = np.array([0.5,0.5])
        # encoded firerates for input - also used for decoding output
        self.zero_fr = zero_fr
        self.one_fr = one_fr
        #input v is will likely be removed but it's kept in for now
        self.input_x = input_x
        self.input_y = input_y
        self.post = lif(rm=10, cm=10, time = time, timestep = 1.0/one_fr, inputv = 0, vt=1, vr = 0)
        
    ''' 
    inputs is an array of tuples: (input_x, input_y, label)
    Using oja's rule and rough logic is that for each input we run a sim and then
    '''   
    def train(self, inputs):
        #using oja's rule here
        # teacher is effectively just going to self.post and yeeting in current
        pass

    # basically just our forward pass w prediction for AND
    def sim(self):
        for i in range(len(self.post.neuron.spikes)):
            # spike in x neuron if input_x == 1 otherwise 0
            self.post.check_spike(self.input_x*self.w[0], i) if self.input_x == 1 else self.post.check_spike(0, i)
            # spike in x neuron if input_y == 1 otherwise 0
            self.post.check_spike(self.input_y*self.w[0], i) if self.input_y == 1 else self.post.check_spike(0, i) 
        # checking if firerate of post == firerate of 1
        if round(float(np.count_nonzeros(self.post.neuron.spikes)/self.post.neuron.time)) == self.one_fr:
            return 1
        else:
            return 0

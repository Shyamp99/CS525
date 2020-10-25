import numpy as np
import plotly as pt
import plotly.graph_objects as go
# import matplotlib as plt
import math

class neuron:
    def __init__(self, time = 100, step = .5):
        #all time units are milliseconds
        self.time = time
        self.step = step

        # so simulate, i value in this arr is considered to be 1 step
        # so like element 0 is t = 0 and element n is t = n*step
        self.time_arr = np.arange(0, time+1, step)
        # membrane potential for a given time: it has 1:1 correspondence with time_arr
        self.vm = np.zeros(int(time/step)+1)
        # tells us when spikes via a 1
        self.spikes = np.zeros(int(time/step)+1)
        # just an array to keep track of our voltages to make it easier to plot
        self.v = np.zeros(int(time/step)+1)
        

    #using plotly to make the graph cuz it's so much fucking better than matplotlib
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
    
    def current_voltage(self, input, step, prev):
        # we are just solving for dv here
        # return self.rm*input - self.tau*input*step
        # print(((-self.neuron.vm[i-1] + input*self.rm) / self.tau)*step)
        return ((-prev + input*self.rm) / self.tau)

    # flip is the time when we want to up voltage: refers to index in time_arr
    # flip end is obv the index at which we want to stop our current and set our current to 0
    def simulate(self, flip, flip_end):
        step = self.neuron.step
        for i in range(1, len(self.neuron.vm)):
            prev = self.neuron.vm[i-1] if self.neuron.vm[i-1]<self.vt else self.vr
            if i < flip:
                mp = prev+self.current_voltage(self.input1, step, prev)
                self.neuron.v[i] = self.input1
            elif (i>=flip and i <= flip_end):
                mp = prev+self.current_voltage(self.input2, step, prev)
                self.neuron.v[i] = self.input2
            elif i > flip_end:
                mp = prev+self.current_voltage(self.vr, step, prev)
                self.neuron.v[i] = self.vr
            if mp >= self.vt:
                self.neuron.spikes[i] = 1
            self.neuron.vm[i] = mp
            
class izhikevich:
    # vm is array of membrane potentials at time t (usually starts at -65mv), vt is threshold (usually +30mv), u is membrane recovery variable (i think it's initially v*b but not sure)
    # a = time scale of recovery var u - smaller - slower recovery, typically 0.02
    # b = sensitivity of recovery var u - greater  results in "Greater values couple and more strongly resulting in possible subthreshold oscillations and low-threshold spiking dynamics
    # c = reset membrane potential - usually -65 mv
    # d = after spike reset of u (represents slow high threshold NA+ and K+ conductance) - usually 2
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 2, vt = 30):
        self.neuron = neuron()
        self.vt = vt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.u = c*b
        self.neuron.vm[0] = c

    def get_du(self, v):
        return self.a*(self.b*v-self.u)

    #What i'm missing is how dv and du play together
    def simulate(self, input_v):
        reset = False
        for i in range(1, len(self.neuron.vm)):
            #I'm not sure if this is correct, i need to check
            if reset:
                self.neuron.vm[i] = self.c
                #this one i'm not sure if we use the voltrage from the last spike or the reset voltage
                #i'm p sure it's that you use v = c
                self.u = self.a*(self.b*self.neuron.vm[i]-self.u)
                #it equals 30 because when we plot the neuron voltage will be 30 at spike and we want the spike lineplot to match
                self.neuron.spikes[i] = 30
                continue

            curr_v = 0.04*self.neuron.vm[i-1]**2 + 5*self.neuron.vm[i-1] + 140 - self.u + input_v
            self.u = self.a*(self.b*self.neuron.vm[i-1]-self.u)
            # once neuron hits 30 we spike and then set v to c, u+=d
            if curr_v >= 30:
                reset = True
                self.u += self.d
                self.neuron.spikes[i] = 30
            self.neuron.vm[i] = curr_v

            

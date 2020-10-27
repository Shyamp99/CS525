import numpy as np
import plotly as pt
import plotly.graph_objects as go
# import matplotlib as plt
import math

class neuron:
    def __init__(self, vt, time = 100, step = .5):
        #all time units are milliseconds
        self.time = time
        self.step = step
        #our spike threshold
        self.vt = vt
        # so simulate, i value in this arr is considered to be 1 step
        # so like element 0 is t = 0 and element n is t = n*step
        self.time_arr = np.arange(0, time+1, step)
        # membrane potential for a given time: it has 1:1 correspondence with time_arr
        self.vm = None
        # tells us when spikes via a 1
        self.spikes = np.zeros(int(time/step)+1)
        # just an array to keep track of our voltages to make it easier to plot
        self.v = np.zeros(int(time/step)+1)
        
        

    #using plotly to make the graph cuz it's so much fucking better than matplotlib
    def plot_graph(self, title):
        vt_line = self.vt*np.ones(int(self.time/self.step)+1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_arr, y=vt_line, mode='lines', name='Spike Threshold', line=dict(color="blue", dash='dot')))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.vm, mode='lines', name='Membrane Potential'))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.v, mode='lines', name='Input Voltage'))
        fig.update_layout(
            title = title,
            xaxis_title="Time (ms)",
            yaxis_title="Voltage (mv)"
        )
        fig.show()
        
class lif:
    # rm = resitence, cm = capcitence, input 1 and 2 are our input voltages
    # vt is our threshold voltage and vr is our reset voltage
    def __init__(self, rm, cm, input1 = .2, input2 = .5, vt=1, vr = 0):
        self.neuron = neuron(vt = vt)
        self.neuron.vm = np.zeros(int(self.neuron.time/self.neuron.step)+1)
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
    # d = after spike reset of u (represents slow high threshold NA+ and K+ conductance) - usually 8 in regular spiking
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 8, vt = 30):
        self.neuron = neuron(vt = vt, time = 500)
        self.neuron.vm = -65*np.ones(int(self.neuron.time/self.neuron.step)+1)
        self.vt = vt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.u = np.zeros(int(self.neuron.time/self.neuron.step)+1)
        self.u[0] = self.b*c
        self.neuron.vm[0] = c

    #What i'm missing is how dv and du play together
    #i'm p sure my logic is sound for this but check it over
    def simulate(self, input_v, input_v2, flip, flip_end):
        k = 0
        inputs = (input_v, input_v2, 0)
        reset = False
        temp = 0
        # get_mp = lambda i: self.neuron.vm[i-1]+self.neuron.step*( 0.04*self.neuron.vm[i-1]**2 + 5*self.neuron.vm[i-1] + 140 - self.u[i-1] + inputs[k])
        # get_u = lambda i: self.u[i-1]+self.neuron.step*(self.a*(self.b*curr_v-self.u[i-1]))
        for i in range(1, len(self.neuron.vm)):
            # print(inputs[k])
            if i == flip:
                k = 1
            elif i == flip_end:
                k =2
            curr_v = self.neuron.vm[i-1]+self.neuron.step*( 0.04*self.neuron.vm[i-1]**2 + 5*self.neuron.vm[i-1] + 140 - self.u[i-1] + inputs[k])
            # print(curr_v)
            self.u[i] = self.u[i-1]+(self.a*(self.b*curr_v-self.u[i-1]))
            # once neuron hits 30 we spike and then set v to c, u+=d
            if reset:
                self.neuron.vm[i-1] = temp
                reset = False
            if curr_v > self.vt:
                temp = curr_v
                self.neuron.vm[i] = self.c
                # print(self.c)
                self.u[i] += self.d
                #30 is a placeholder i'll adjust the value according tothe final graph
                self.neuron.spikes[i] = self.vt
                self.neuron.v[i] = inputs[k]
                reset = True
                continue
            self.neuron.vm[i] = curr_v
            self.neuron.v[i] = inputs[k]

            

import numpy as np
import plotly as pt
import plotly.graph_objects as go
# import matplotlib as plt
import math
from scipy.integrate import odeint

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
            fig.add_trace(go.Scatter(x=self.time_arr, y=vt_line, mode='lines', name='Spike Threshold (mv)', line=dict(color="blue", dash='dot')))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.vm, mode='lines', name='Membrane Potential (mv)'))
        fig.add_trace(go.Scatter(x=self.time_arr, y=self.a, mode='lines', name='Input Current (a)'))
        fig.update_layout(
            title = title,
            xaxis_title="Time (ms)",
            yaxis_title="Value"
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

    # we are just solving for dv here
    def current_voltage(self, input, step, prev):
        return ((-prev + input*self.rm) / self.tau)

    # flip is the time when we want to up voltage: refers to index in time_arr
    # flip end is obv the index at which we want to stop our current and set our current to 0
    def simulate(self, flip, flip_end):
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
            
class izhikevich:
    # vm is array of membrane potentials at time t (usually starts at -65mv), vt is threshold (usually +30mv), u is membrane recovery variable
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

    # the two input_a parameters are the two input amperages
    # flip, flip_end represents the portion in time_arr where we want ot up the amperage
    # after flip end the input current is 0
    def simulate(self, input_a, input_a2, flip, flip_end):
        k = 0
        inputs = (input_a, input_a2, 0)
        reset = False
        temp = 0
        # get_mp = lambda i: self.neuron.vm[i-1]+self.neuron.step*( 0.04*self.neuron.vm[i-1]**2 + 5*self.neuron.vm[i-1] + 140 - self.u[i-1] + inputs[k])
        # get_u = lambda i: self.u[i-1]+self.neuron.step*(self.a*(self.b*curr_v-self.u[i-1]))
        for i in range(1, len(self.neuron.vm)):
            if i == flip:
                k = 1
            elif i == flip_end:
                k =2
            curr_v = self.neuron.vm[i-1]+self.neuron.step*( 0.04*self.neuron.vm[i-1]**2 + 5*self.neuron.vm[i-1] + 140 - self.u[i-1] + inputs[k])
            self.u[i] = self.u[i-1]+(self.a*(self.b*curr_v-self.u[i-1]))
            if reset:
                self.neuron.vm[i-1] = temp
                reset = False
            if curr_v > self.vt:
                temp = curr_v
                self.neuron.vm[i] = self.c
                self.u[i] += self.d
                self.neuron.spikes[i] = self.vt
                self.neuron.a[i] = inputs[k]
                reset = True
                continue
            self.neuron.vm[i] = curr_v
            self.neuron.a[i] = inputs[k]

class hodgkinhuxley:
    
    """
    Constants
    """
    # capacitance of membrance
    c_m = 1.0
    
    # Equilibrium potentials for each ion
    v_Na = -115
    v_K = 12
    v_L = -10.613
    #v_Na = 50.0
    #v_K = -77
    #v_L = -54.387
    
    # Conductances for each ion
    g_Na = 120
    g_K = 36
    g_L = 0.3
    
    
    def __init__(self, vt=6, t=100, step=.5):
        self.neuron = neuron(vt=vt, time=t, step=step)
        self.neuron.vm = np.zeros(int(self.neuron.time/self.neuron.step)+1)
        print(self.neuron.time_arr)
        
    def simulate(self):
        X = odeint( self.dvdt, [-65, 0.05, 0.6, 0.32], self.neuron.time_arr,args=(self,),mxstep=500 )
        self.neuron.vm = X[:,0]
        return X
    
    
    """
    Calculating rate of gates opening in Potassium channels
    """
    def a_n(self, V):
        return 0.01 * ( ( 10 - V ) / ( np.exp( (10 - V) / 10 ) - 1 ) )
    """
    Calculating rate of gates closing in Potassium channels
    """
    def b_n(self, V):
        return 0.125 * ( np.exp( (-1) * V / 80 )  )
    
    """
    Calculating rate of the fast gates opening in Sodium channels
    """
    def a_m(self, V):
        return 0.1 * ( ( 25 - V ) / ( np.exp( ( 25 - V ) / 10 ) - 1 ) )
    """
    Calculating rate of the fast gates closing in Sodium channels
    """
    def b_m(self, V):
        return 4 * ( np.exp(  (-1) * V / 18  ) )
    
    """
    Calculating rate of the slow gates opening in Sodium Channels
    """
    def a_h(self, V):
        return 0.07 * ( np.exp( (-1) * V / 20 ) )
    """
    Calculating rate of the slow gates closing in Sodium Channels
    """
    def b_h(self, V):
        return 1 / ( np.exp( ( 30 - V ) / 10 ) + 1 )
    
    """
    Returning the input current based on time
    """
    def I_in(self, t):
        """
        if t > 75:
            return 0
        elif t > 55:
            return 35
        elif t > 35:
            return 0
        elif t > 15:
            return 10
        else:
            return 0
       """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)
    """
    Calculate the leaking current
    """
    def I_leak(self, V):
        return self.g_L * ( V - self.v_L )
    """
    Calcuate the sodium current
    """
    def I_Na(self, m, h, V):
        return self.g_Na * m**3 * h * ( V - self.v_Na )
    """
    Calculate the potassium current
    """
    def I_K(self, n, V):
        return self.g_K * n**4 * ( V - self.v_K )
    """
    Calculate dv/dt at the given time
    """
    @staticmethod
    def dvdt(X, t, self):
        V, m, h, n = X
        dVdt = (self.I_in(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_leak(V)) / self.c_m
        dmdt = self.a_m(V)*(1.0-m) - self.b_m(V)*m
        dhdt = self.a_h(V)*(1.0-h) - self.b_h(V)*h
        dndt = self.a_n(V)*(1.0-n) - self.b_n(V)*n
        return dVdt, dmdt, dhdt, dndt

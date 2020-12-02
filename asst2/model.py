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
        self.vt = vt
        self.vr = vr
        self.mp = 0

    # we are just solving for dv here
    # for updating mp with each iteration
    def update_voltage(self, input_curr):
        if input_curr == 0:
            return 0
        # else:
        #     print(input_curr)
        return ((-1*self.mp + input_curr*self.rm) / self.tau)

    # basically called if teacher nueron or input neuron spikes 
    def check_spike(self, input_current, time, train = False, label = None):
        if train:
            # updating voltage based on formula for LIF/from last assignment
            self.mp += self.update_voltage(input_current)

            if label == 1:
                self.mp = self.vr
                self.neuron.spikes[time] = 1
            else:
                # updating voltage based on formula for LIF/from last assignment
                self.mp += self.update_voltage(input_current)

                #if spike we handle reset
                if self.mp >= self.vt:
                    self.mp = self.vr
                    self.neuron.spikes[time] = 1
        else:
            # updating voltage based on formula for LIF/from last assignment
            temp = self.update_voltage(input_current)

            # debug print
            # print("input_current = ", input_current, " temp = ", temp, " mp = ", self.mp)

            self.mp += temp

            # for debugging
            # if self.mp == 0:
                # print("check_spike input current = ", input_current)
                # print("check_spike temp = ", temp)
                # print("check_spike mp = ", self.mp, '\n')

            # if spike we handle reset
            if self.mp >= self.vt:
                self.mp = self.vr
                self.neuron.spikes[time] = 1

            # for debugging
            # else:
            #     print(self.mp)


class AND_model:
    '''
    input_x and y are the 0 or 1 values, zero_fr and one_fr are the firing rates we are encoding 0 and 1 as
    post is our post synaptic neuron
    '''
    def __init__(self, target_fr = 4, zero_fr = 1, one_fr = 4, time = 5):
        #initialize both to 0.5 because 1/number of weights since we're using oja's rule
        self.w = np.array([0.5,0.5])

        # encoded firerates for input - also used for decoding output
        self.zero_fr = zero_fr
        self.one_fr = one_fr
        self.target_fr = target_fr
        self.post = lif(rm=5, cm=0.75, time = time, timestep = 1.0/one_fr, inputv = 0, vt=1, vr = 0)
        
    ''' 
    inputs is an array of tuples: (input_x, input_y, label)
    Using oja's rule and rough logic is that for each input we run a sim and then
    '''   
    def train(self, inputs, alpha = 0.0001):
        # lambda to calculate dw (weight change) in oja's
        oja = lambda in_fr, out_fr, alpha, weight: alpha*(in_fr*out_fr-weight*out_fr**2)

        # going through the inputs in a batch
        for inp in inputs:
            
            input_x = inp[0]
            x_fr = self.one_fr if input_x == 1 else self.zero_fr
            input_y = inp[1]
            y_fr = self.one_fr if input_y == 1 else self.zero_fr
            label = inp[2]

            # running the simulation for training
            # logic: calculate how input affects post and then run oja's for each specific weight
            for i in range(len(self.post.neuron.spikes)):
                # for input neuron x
                if input_x == 1:
                    self.post.check_spike(self.w[0], i, train = True, label = label)
                else:
                    # checking to see if we are at whole second since zero_fr = 1 hz
                    if i % self.post.neuron.time == 0:
                        self.post.check_spike(self.w[0], i, train = True, label = label)
                    else:
                        self.post.check_spike(0, i, train = True, label = label)

                curr_fr = round(float(np.count_nonzero(self.post.neuron.spikes)/((i+1)*self.post.neuron.timestep)))
                dw = oja(x_fr, curr_fr, alpha, self.w[0])
                self.w[0] += dw
                self.w[1] -= dw
                
                #for input neuron y
                if input_y == 1:
                    self.post.check_spike(self.w[1], i, train = True, label = label)
                else:
                    # checking to see if we are at whole second since zero_fr = 1 hz
                    if i % self.post.neuron.time == 0:
                        self.post.check_spike(self.w[1], i, train = True, label = label)
                    else:
                        self.post.check_spike(0, i, train = True, label = label)
                curr_fr = round(float(np.count_nonzero(self.post.neuron.spikes)/((i+1)*self.post.neuron.timestep)))
                dw = oja(y_fr, curr_fr, alpha, self.w[1])
                self.w[1] += dw
                self.w[0] -= dw
            self.post.mp = 0
            self.post.neuron.spikes = np.zeros(int(self.post.neuron.time/self.post.neuron.timestep)+1)

    # basically just our forward pass w prediction for AND
    def sim(self, input_x, input_y):
        for i in range(len(self.post.neuron.spikes)):
            # spike in x neuron if input_x == 1 otherwise 0 then same logic for input y neuron
            if input_x == 1:
                self.post.check_spike(input_x*self.w[0], i)
            else:
                # checking to see if we are at whole second since zero_fr = 1 hz
                if i % self.post.neuron.time == 0:
                    self.post.check_spike(self.w[0], i)
                else:
                    self.post.check_spike(0, i)

            #for input neuron y
            if input_y == 1:
                self.post.check_spike(self.w[1], i)
            else:
                # checking to see if we are at whole second since zero_fr = 1 hz
                if i % self.post.neuron.time == 0:
                    self.post.check_spike(self.w[1], i)
                else:
                    self.post.check_spike(0, i)
        
        # for debugging
        print(self.post.neuron.spikes)
        print("calculated firing rate: ", np.count_nonzero(self.post.neuron.spikes)/self.post.neuron.time, '\n\n')

        # checking if firerate of post == firerate of 1
        if round(float(np.count_nonzero(self.post.neuron.spikes)/self.post.neuron.time)) >= self.target_fr:
            self.post.mp = 0
            self.post.neuron.spikes = np.zeros(int(self.post.neuron.time/self.post.neuron.timestep)+1)
            return 1
        else:
            self.post.mp = 0
            self.post.neuron.spikes = np.zeros(int(self.post.neuron.time/self.post.neuron.timestep)+1)
            return 0

import numpy as np
import plotly as pt
import plotly.graph_objects as go
import matplotlib as plt
import matplotlib.pyplot as plot
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
        
    def reset_spikes(self):
        self.spikes = np.zeros(int(self.time/self.timestep)+1)

    def calc_fr(self):
        return round(float(np.count_nonzero(self.spikes)/(self.time))) 

    def prep_plot(self, input_x, input_y):
        #preparing the plot
        title = 'Raster plot for X = ' + str(input_x) + ' and Y = '+ str(input_y) 
        print('input_x: ', type(input_x))
        if input_x == 1:
            x_spike = np.ones(len(self.spikes))
            print(x_spike)
        else:
            x_spike = np.zeros(len(self.spikes))
            for i in range(len(self.spikes)):
                if i!= 0 and i % self.time == 0:
                    x_spike[i] = 1
        if input_y == 1:
            y_spike = np.ones(len(self.spikes))
        else:
            y_spike = np.zeros(len(self.spikes))
            for i in range(len(self.spikes)):
                if i!= 0 and i % self.time == 0:
                    y_spike[i] = 1
        self.plot_graph(title, x_spike, y_spike)

    # x_spike and y_spike are the respective spiketrains for the and_model
    def plot_graph(self, title, x_spike, y_spike):

        post_spike = self.spikes
        totalTime=self.time
        timeStepSize=self.timestep

        #temp = np.arange(0,1,1/len(x_spike))
        temp = np.arange(0, totalTime + timeStepSize, timeStepSize)
        # print("temp: {}".format(temp))
        #print("totalTime: {} timeStepSize: {} numTimeSteps: {}".format(totalTime, timeStepSize, numTimeSteps))

        # Create 3 x numTimeSteps size 2d array filled with zeros
        neuralData = []
        neuralData.append(x_spike*temp)
        neuralData.append(y_spike*temp)
        neuralData.append(post_spike*temp)

        # debug print
        # print("Neural data: {}".format(neuralData))

        # Draw a spike raster plot
        colorCodes = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]])

        lineSize = [0.75, 0.75, 0.75]                                  
        plot.eventplot(neuralData, color=colorCodes, linelengths = lineSize)     
        plot.title(title)

        # Give x axis label for the spike raster plot
        plot.xlabel('Timesteps')

        # Give y axis label for the spike raster plot
        plot.ylabel('Neuron')
        plot.yticks(np.arange(3), ['Post', 'Y', 'X'])
        # print(temp*x_spike)

        # Display the spike raster plot
        plot.show()

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
        self.total_current = 0

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
        #if time == 0:
        #    return
        if train:
            # updating voltage based on formula for LIF/from last assignment
            temp =  self.update_voltage(input_current)
            self.mp += temp
            self.total_current += temp

            if label == 1:
                self.mp = self.vr
                self.neuron.spikes[time] = 1
            if self.mp >= self.vt:
                    self.mp = self.vr
                    self.neuron.spikes[time] = 1
            else:
                # updating voltage based on formula for LIF/from last assignment
                temp =  self.update_voltage(input_current)
                self.mp += temp
                self.total_current += temp

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
            self.total_current += temp

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
        # print("Spiketrain for Post: ", self.post.neuron.spikes)
        # print("calculated firing rate: ", np.count_nonzero(self.post.neuron.spikes)/self.post.neuron.time, '\n\n')


class num_model:
    
    def __init__(self, time, timestep = 1/16, encoded_frs = np.arange(17), output_neuron_count = 10):
        self.time = float(time)
        self.timestep = float(timestep)
        self.encoded_frs = encoded_frs
        self.weights = [[0]*64 for i in range(output_neuron_count)]
        self.output_neurons = [lif(rm=5, cm=0.75, time = time, timestep = timestep, inputv = 0, vt=1, vr = 0) for i in range(output_neuron_count)]
    
    def plot_accuracies(self, results):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = np.arange(1, len(results)+1), y = results,mode = 'lines' name = 'Validation Accuracy (%)'))
        fig.update_voltage(
            title = 'Validation Accuracies for each Epoch',
            xaxis_title="Epoch Number",
            yaxis_title="Accuracy (%)"
        )
        fig.show()

    '''
    
    '''
    def poisson_encoding(self, image):
        # normalize the image
        normalized_image = np.divide( image, np.amax(image) )
        
        # Generate uniform random numbers
        X = np.random.uniform( 0, 1, 64 )
        
        # Using both determine if specific pixel / neuron fires
        dt = 1
        fire = np.zeros( 64 )
        for idx in range( len(normalized_image) ):
            if normalized_image[idx] * dt > X[idx]:
                fire[idx] = 1
        return fire

    
        # full_reset implies a full reset of weights
    def reset_nn(self, full_reset = False):
        if full_reset:
            self.weights = [[0.1]*64 for i in range(len(self.weights))]
        for i in range(len(self.output_neurons)):
            self.output_neurons[i].total_current = 0
            self.output_neurons[i].mp = 0
            self.output_neurons[i].neuron.reset_spikes()

    # i = index of neuron weight that is increased, out_index = output neuron for that weight
    def oja(self, i, out_index, in_fr, out_fr, alpha = 0.0001):
        # print(i)
        dw = alpha*(in_fr*out_fr-self.weights[out_index][i]*out_fr**2)
        self.weights[out_index][i] += dw
        for index in range(len(self.weights)):
            if index != i:
                self.weights[index][out_index] -= dw/(len(self.weights)-1)

    def print_weights(self):
        for o_neuron in range(len( self.weights )):
            print('o_neuron {}:'.format(o_neuron))
            for i_neuron in range( 64 ):
                print(self.weights[o_neuron][i_neuron], end=' ')
            print()
               
     #inputs is tuple of (flattened image arr, label number as int)
    def train(self, X, y, a):
        '''
        Params:
            X: list of arrays that represent the 8x8 handwritten image
            y: list of ints that match the value of the 8x8 handwritten image
            a: lr for hebbian rule
        Return:
            none
        '''
        #print('length of X: {}'.format(len(X)))
        # loop through all of the training data
        for t in range( len(X) ):
            # get which input neurons are firing
            fire_map = self.poisson_encoding( X[t] )
            
            # loop through each neuron in the fire_map and update weights
            for neuron_number in range( len( fire_map ) ):
                # check if neuron is not firing
                if fire_map[ neuron_number ] == 0:
                    continue
                   
                # we now update weight between this input neuron and the
                # correct output neuron
                # change based on LR, Pixel Intensity, and whether output spikes
                self.weights[y[t]][neuron_number] +=  a * X[t][neuron_number] * 1
        
        
        self.reset_nn()

    def test(self, X_test, y_test):
        num_correct = 0
        for t in range( len(X_test) ):
            #print("t: {}, Simulated: {}, Expected: {}".format(t, test.sim( X_test[t] ), y_test[t] ))
            if self.sim( X_test[t] ) == y_test[t]:
                num_correct += 1
        return num_correct / len(X_test)
    
    def validate(self, X_val, y_val):
        num_correct = 0
        for t in range( len(X_val) ):
            #print("t: {}, Simulated: {}, Expected: {}".format(t, test.sim( X_test[t] ), y_test[t] ))
            if self.sim( X_val[t] ) == y_val[t]:
                num_correct += 1
        return num_correct / len(X_val)
        
    def sim(self, image_arr):
        fire_map = self.poisson_encoding( image_arr )
        for time in range(len(self.output_neurons[0].neuron.spikes)):
            # output neurons
            for out_n in range(len(self.output_neurons)):
                #input neurons
                curr_out = self.output_neurons[out_n]
                for in_n in range(64):
                    # in_fr = image_arr[in_n]
                    curr_weight = self.weights[out_n][in_n]
                    # input neuron is spiking
                    if fire_map[in_n] == 1:
                        curr_out.check_spike(curr_weight, time, train = False)
                         
                    # input neuron doesn't fire or fr = 0
                    else:
                        curr_out.check_spike(0, time)

        # getting index of neuron w highest firerate      
        max_tc = 0
        max_neuron = -1
        for i in range(len(self.output_neurons)):
            tc = self.output_neurons[i].total_current
            #print(tc)
            if tc > max_tc:
                max_tc = tc
                max_neuron = i
        self.reset_nn()
        
        return max_neuron
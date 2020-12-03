import matplotlib.pyplot as plot
import numpy as np
 
def draw_plot(x_spike, y_spike, post_spike, totalTime, timeStepSize):
	# Set x-axis as time intervals 
	numTimeSteps = totalTime / timeStepSize
	print("totalTime: {} timeStepSize: {} numTimeSteps: {}".format(totalTime, timeStepSize, numTimeSteps))

	# Create 3 x numTimeSteps size 2d array filled with zeros
	neuralData = [[0]*int(numTimeSteps)]*3

	# Change values in 2d array to match given data
	for x in range(len(x_spike)):
		neuralData[0][x] = x_spike[x]
		neuralData[1][x] = y_spike[x]
		neuralData[2][x] = post_synaptic_spike[x]
	print("Neural data: {}".format(neuralData))

	# Draw a spike raster plot
	colorCodes = np.array([[0, 0, 0],
				[1, 0, 0],
				[0, 1, 0]])
	lineSize = [1, 1, 1]                                  
	plot.eventplot(neuralData, color=colorCodes, linelengths = lineSize)     
	plot.title('Spike raster plot')

	# Give x axis label for the spike raster plot
	plot.xlabel('Timesteps')

	# Give y axis label for the spike raster plot
	plot.ylabel('Neuron')

	# Display the spike raster plot
	plot.show()

x_spike = [0,1,0,0,0,1]
y_spike = [1,1,0,0,1,1]
post_synaptic_spike = [0,1,0,0,0,1]
totalTime = 5
timeStepSize = 0.25
draw_plot(x_spike, y_spike, post_synaptic_spike, totalTime, timeStepSize)

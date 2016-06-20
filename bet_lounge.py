# -*- coding: utf-8 -*-
"""
Example of use singe layer perceptron(newp)
===========================================

Train with Delta rule

"""

import neurolab as nl

# Logical &
input = [[1, 4, 1, 2, 3, 0], [5, 0, 1, 4, 1, 0], [2, 3, 1, 3, 2, 0], [5, 0, 1, 1, 5, 0], [3, 3, 1, 4, 1, 0],
         [2, 5, 1, 5, 0, 0], [5, 0, 1, 3, 4, 0], [1, 6, 1, 4, 1, 0], [3, 5, 1, 1, 7, 0], [4, 1, 1, 2, 5, 0],
	[1, 7, 1, 2, 5, 0], [5, 0 ,1, 4, 1, 0], [2, 5, 1, 3, 5, 0], [5, 0, 1, 1, 7, 0], [3, 5, 1, 4, 1, 0], 
	[2, 6, 1, 5, 0, 0], [5, 0, 1, 3, 6, 0], [1, 8, 1, 4, 1, 0], [1, 9, 1, 3, 7, 0], [4, 1, 1, 2, 7, 0]]
target = [[1], [0], [1], [2], [1], [1], [2], [1], [0], [2], [0], [0], [1], [2], [1], [1], [2], [1], [0], [2]]

# Create net with 2 inputs and 1 neuron
net = nl.net.newff([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]],[10, 40, 1] )

# train with delta rule
# see net.trainf
error = net.train(input, target, epochs=100, show=10, lr=1)

# Plot results
import pylab as pl

pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()

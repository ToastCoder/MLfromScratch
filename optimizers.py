# ML FROM SCRATCH

# ML OPTIMIZERS, LOSS FUNCTIONS AND ALGORITHMS IMPLEMENTED FROM SCRATCH

# FILE NAME: OPTIMIZERS.PY

# IMPORTING REQUIRED LIBRARIES
import numpy as np

#OPTIMIZER CLASS DEFINITION
class Optimizers:
    # CONSTRUCTOR
    def __init__(self,num_weights,learning_rate,momentum_coeff):
        self.previous_updates = [0] * num_weights
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.cache = [0] * num_weights
        self.m = [0] * num_weights
        self.v = [0] * num_weights
        self.t = 1

        # STOCHASTIC GRADIENT DESCENT FUNCTION
        def sgd(self,weights,gradients):
            
            updated_weights = []

            for weights, gradients in zip(weights, gradients):
                delta = self.learning_rate * gradients
                weights -= delta
                updated_weights.append(weights)

            return updated_weights

        # SGD MOMENTUM FUNCTION
        def sgd_momentum(self,weights,gradients):

            updated_weights = []
            prevs = []

            for weights,gradients,prev_update in zip(weights, gradients, self.previous_updates):
                delta = self.learning_rate * gradients - self.momentum_coeff * prev_update
                weights -= delta

                prevs.append(delta)
                updated_weights.append(weights)
                self.previous_udpates = prevs
            
            return updated_weights
        
        # ADAGRAD FUNCTION
        def adagrad(self,weights,gradients):

            updated_weights = []
            for i, (weights, gradients) in enumerate(zip(weights, gradients)):
                self.cache[i] += gradients ** 2
                weights += -self.learning_rate * gradients / (np.sqrt(self.cache[i])+1e-6)

                updated_weights.append(weights)
            
            return updated_weights
        
        # RMS PROP FUNCTION
        def rmsprop(self, weights, gradients, decay_rate = 0.9):

            updated_weights = []

            for i,(weights, gradients) in enumerate(zip(weights, gradients)):
                self.cache[i] = decay_rate * self.cache[i] + (1 - decay_rate) * gradients ** 2
                weights += -self.learning_rate * gradients / (np.sqrt(self.cache[i])+1e-6)

                updated_weights.append(weights)

            return updated_weights
        
        # ADAM FUNCTION
        def adam(self, weights, gradients, beta1 = 0.9, beta2 = 0.999):

            updated_weights = []

            for i, (weights, gradients) in enumerate(zip(weights, gradients)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * gradients ** 2
                m_corrected = self.m[i] / (1 - beta1 ** self.t)
                v_corrected = self.v[i] / (1 - beta2 ** self.t)
                weights += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)

                updated_weights.append(weights)
                self.t += 1

            return updated_weights
        



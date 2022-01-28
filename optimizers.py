# ML FROM SCRATCH

# ML OPTIMIZERS, LOSS FUNCTIONS AND ALGORITHMS IMPLEMENTED FROM SCRATCH

# FILE NAME: OPTIMIZERS.PY

#OPTIMIZER CLASS DEFINITION
class Optimizers:
    # CONSTRUCTOR
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

        # STOCHASTIC GRADIENT DESCENT FUNCTION
        def sgd(self,weights,gradients):
            
            updated_weights = []

            for weights, gradients in zip(weights, gradients):
                delta = self.learning_rate * gradients
                weights -= delta
                updated_weights.append(weights)

            return updated_weights


import numpy as np
import random
import pandas as pd
import utils

log = utils.get_logger()

class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=30,
                 optimizer='sgd', epsilon=1e-8):
        self.model = model

        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.optimizer = optimizer
        if self.optimizer == 'sgd':
            log.info("Using sgd..")
        elif self.optimizer == 'adagrad':
            log.info("Using adagrad...")
            epsilon = epsilon
            self.gradt = [epsilon + np.zeros(W.shape) for W in self.model.stack]
        else:
            raise ValueError("Invalid optimizer")

        self.costt = []
        self.expcost = []

    def run(self,trees):
        """
        Runs stochastic gradient descent with model as objective.
        """
        df_total = pd.DataFrame()
        correct = total = cost_total = 0.0
        iterations_per_epoch = 0
        
        m = len(trees)

        # randomly shuffle data
        random.shuffle(trees)

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1
            iterations_per_epoch += 1

            mb_data = trees[i:i+self.minibatch]
               
            cost,grad, df, corr, tot = self.model.costAndGrad(mb_data)
            df_total = df_total.append(df)
            correct += corr
            total += tot
            cost_total += cost

            # compute exponentially weighted cost
            if np.isfinite(cost):
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)

            if self.optimizer == 'sgd':
                update = grad
                scale = -self.alpha

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2 
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in dL.iterkeys():
                    dLt[:,j] = dLt[:,j] + dL[j]**2
                    dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.alpha


            # update params
            self.model.updateParams(scale,update,log=False)

            self.costt.append(cost)
            if self.it%1 == 0:
                # TODO: add metrics
                log.info("Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1]))

        return df_total, self.model, cost_total/iterations_per_epoch, correct/float(total)
            

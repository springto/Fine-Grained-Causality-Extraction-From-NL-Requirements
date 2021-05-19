from imp import reload

import numpy as np
import collections
import tree
import pandas as pd
import nltk
import utils
import fasttext
import fasttext.util

np.seterr(over='raise',under='raise')
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#log = utils.get_logger()

nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
reversedWordMap = tree.loadReversedWordMap()
fasttext.util.download_model('en', if_exists = 'ignore')
ft_pos = fasttext.load_model('cc.en.300.bin')
#ft_word = fasttext.load_model('cc.en.300.bin')


class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        #fasttext.util.reduce_model(ft_word, self.wvecDim/2)
        fasttext.util.reduce_model(ft_pos, self.wvecDim)

    def initParams(self):

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.createWordVectors(self.wvecDim)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))


    def createWordVectors(self, wvecDim):
        """
        Each word in the word list gets a random embedding and a pos tag
        these two vectors will be concatenated and set in the corresponding column in L
        :param wvecDim:
        :return:
        """
        # reversedWordMap (2: 'the')
        for number, text in reversedWordMap.items():
            #[(test, 'CC')]
            tokenized_text = nltk.word_tokenize(text)
            pos_tag = nltk.pos_tag(tokenized_text)
            pos_string = ""
            for (text, tag) in pos_tag:
                pos_string += tag + "_"
            pos_string = pos_string[:-1]
            # string = ADJ_PP
            # text = it's
            # it's --> [1, 4, 8, 9]
            #text_embedding = ft_word.get_word_vector(text)
            # ADJ_PP --> [1, 87, 8, 345]
            pos_embedding = ft_pos.get_word_vector(pos_string)
            # concatenate
            #combined_embedding = np.concatenate([text_embedding, pos_embedding], axis= 0)
            self.L[:, number] = pos_embedding

        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.
        """
        cost = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        total_df = pd.DataFrame()

        self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack
        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,corr,tot, df = self.forwardProp(tree.root)
            cost += c
            correct += corr
            total += tot
            total_df = total_df.append(df, ignore_index = True)
        if test:
            return (1./len(mbdata))*cost,correct,total, total_df

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.V**2)
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dV+self.rho*self.V),
                           scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs], total_df, correct, total


    def printPredictions(self, node, wordmap):
        string = ""
        if node.isLeaf:
            if node.word in reversedWordMap:
                return reversedWordMap[node.word] + " "
            else:
                return "unknowntoken "

            # for wordtext, number in wordmap.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            #     #print((wordtext, number))
            #     #print("word " + str(node.word))
            #     if number == node.word:
            #         #print("This word " + str(node.word) + " is in the wordmap")
            #         return " " + wordtext
        if node.left is not None:
            string += self.printPredictions(node.left, wordmap)
        if node.right is not None:
            string += self.printPredictions(node.right, wordmap)
        return string


    def forwardProp(self,node):
        cost = correct =  total = 0.0
        #columns = ["token", "probability_predicted", "prediction", "probability_correct", "ground_truth"]
        df = pd.DataFrame()

        if node.isLeaf:
            node.hActs = self.L[:,node.word]
            node.fprop = True

        else:
            if not node.left.fprop:
                c,corr,tot, df_new = self.forwardProp(node.left)
                cost += c
                correct += corr
                total += tot
                df = df.append(df_new, ignore_index = True)
            if not node.right.fprop:
                c,corr,tot, df_new = self.forwardProp(node.right)
                cost += c
                correct += corr
                total += tot
                df = df.append(df_new, ignore_index = True)
            # Affine
            lr = np.hstack([node.left.hActs, node.right.hActs])
            node.hActs = np.dot(self.W,lr) + self.b
            node.hActs += np.tensordot(self.V,np.outer(lr,lr),axes=([1,2],[0,1]))
            # Tanh
            node.hActs = np.tanh(node.hActs)

        # Softmax
        node.probs = np.dot(self.Ws,node.hActs) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)

        node.fprop = True

        wordmap = tree.loadWordMap()
        #print("Label number: " + str(np.argmax(node.probs)))
        token = self.printPredictions(node, wordmap)
        prediction = np.argmax(node.probs)
        probability_predicted = node.probs[np.argmax(node.probs)]
        probability_correct = node.probs[node.label]
        ground_truth = node.label

        row = {"token": token.encode('utf-8'), "prediction": prediction, "probability_predicted": probability_predicted,
               "probability_correct": probability_correct, "ground_truth": ground_truth}
        df = df.append(row, ignore_index = True)

        return cost - np.log(node.probs[node.label]), correct + (np.argmax(node.probs)==node.label),total + 1, df


    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        # Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWs += np.outer(deltas,node.hActs)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T,deltas)
        
        if error is not None:
            deltas += error

        deltas *= (1-node.hActs**2)

        # Leaf nodes update word vecs
        if node.isLeaf:
            self.dL[node.word] += deltas
            return

        # Hidden grad
        if not node.isLeaf:
            lr = np.hstack([node.left.hActs, node.right.hActs])
            outer = np.outer(deltas,lr)
            self.dV += (np.outer(lr,lr)[...,None]*deltas).T
            self.dW += outer
            self.db += deltas
            # Error signal to children
            deltas = np.dot(self.W.T, deltas) 
            deltas += np.tensordot(self.V.transpose((0,2,1))+self.V,
                                   outer.T,axes=([1,0],[0,1]))
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                log.info("weight rms=%f -- update rms=%f"%(pRMS,dpRMS))

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):


        cost, grad, df = self.costAndGrad(data)

        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        log.info("Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err))

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                log.info("Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err))


if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 42

    rntn = RNN(wvecDim,outputDim,numW,mbSize=4)
    rntn.initParams()

    mbData = train[:1]
    #cost, grad = rntn.costAndGrad(mbData)

    log.info("Numerical gradient check...")
    rntn.check_grad(mbData)







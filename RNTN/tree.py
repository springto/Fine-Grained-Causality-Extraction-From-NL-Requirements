

import collections
import re
import utils
from random import sample
log = utils.get_logger()
UNK = 'UNK'

class Node:
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        tokens = re.findall(r'\(|\)|[^\(\) ]+', treeString.rstrip("\n"))
        # for toks in treeString.strip().split():
        #     tokens += list(toks)
        self.root = self.parse(tokens)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2 # position after open and label
        countOpen = countClose = 0



        if tokens[split] == self.open and tokens[split+1] != self.close:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open and tokens[split+1] != self.close:
                countOpen += 1
            if tokens[split] == self.close:
                #if tokens[split-1] == "2" and tokens[split-2] != "38":
                if tokens[split - 1] == "2" and tokens[split - 2] != "23":
                    pass
                else:
                    countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1])-1) # zero index labels
        node.parent = parent 

        # leaf Node
        if countOpen == 0:
            #print(tokens[2:-1])
            #print(''.join(tokens[2:-1]).lower())
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)
        return node

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)


def countWords(node, words):
    if node.isLeaf:
        words[node.word] += 1

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]

def loadWordMap():
    import cPickle as pickle
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def loadReversedWordMap():
    import cPickle as pickle
    with open('wordMapReversed.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """
    import cPickle as pickle
    file = 'trees/train.txt'
    log.info("Reading trees..")
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    log.info("Counting words..")
    words = collections.defaultdict(int)
    reversedWords = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMapReversed = dict(zip(xrange(len(words)), words.iterkeys()))
    wordMap[UNK] = len(words) # Add unknown as word

    print("This is the word map")
    print(wordMap)

    print("This is the reversed word map")
    print(wordMapReversed)

    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

    with open('wordMapReversed.bin','w') as fid:
        pickle.dump(wordMapReversed,fid)


def loadTrees(dataSet='train', sample_size=None):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    file = 'trees/%s.txt'%dataSet
    log.info("Reading trees..")
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    if sample_size is not None:
        assert isinstance(sample_size, int) or isinstance(sample_size, float), \
            "The sample size has to be either int (number of elements) or float (fraction of the initial dataset)"
        if isinstance(sample_size, int):
            assert 0 < sample_size <= len(trees), \
                "Wrong sample size; if it's an integer then must be in [1," + len(trees) + "]"
        if isinstance(sample_size, float):
            assert 0 < sample_size <= 1, \
                "Wrong sample size; if it's a float must be in (0, 1]"
            sample_size = int(sample_size * len(trees))
            assert sample_size > 0, "Sample fraction too small"
        if sample_size < len(trees):
            trees = sample(trees, sample_size)

    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees
      
if __name__=='__main__':
    buildWordMap()
    train = loadTrees()




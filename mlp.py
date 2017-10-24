# mlp.py
# -------------

# mlp implementation
import util
PRINT = True

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
    self.inweights = {}
    self.hidweights = {}
    HID_LAY = 10

    for x in range(HID_LAY):
    	self.hidweights[x] = util.Counter()

    for label in legalLabels:
    	self.inweights[label] = util.Counter()


      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):

  	self.features = trainingData[0].keys()

  	import random

	for j in range(HID_LAY):
      for f in self.features:
        self.inweights[j][f] = random.randint(1,6)

    for j in range(len(self.legalLabels)):
      for f in range(HID_LAY):
        self.hidweights[j][f] = random.randint(1,6)




    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."

      correct = 0;
      incorrect = 0;


      for i in range(len(trainingData)):
      	#score calc

        hidscore = util.Counter()
        fscore = util.Counter()
        for j in range(len(self.hidweights)):
        	hidscore[j] = trainingData[i]*self.inweights[j]
      	  	
      	for l in range(len(self.legalLabels)):
      		fscore[l] = 1/(1+pow(math.e,(hidscore*self.hidweights[l])))

      	prediction = fscore.argMax()

      	if prediction != trainingLabels[i]:
      		ErrVecOut = util.Counter()
      		ErrVecOutD = util.Counter()

      		for i in range(len(self.legalLabels)):
      			ErrVecOut[i]=(trainingLabels[i]-prediction)
      			ErrVecOutD[i]= ErrVecOut[i]*(1/(1+pow(math.e,(hidscore*self.hidweights[i]))))*(1-(1/(1+pow(math.e,(hidscore*self.hidweights[i])))))


      		backProp(self,0.30,hidscore,trainingData,ErrVecOut,ErrVecOutD)



    #util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"
      util.raiseNotDefined()
    return guesses

  def backProp(self,lr,hidscore,trainingData,ErrVecOut,ErrVecOutD):
    	for i in range(len(self.legalLabels)):
    		for j in range(len(HID_LAY)):
    			hidweights[i][j]= hidweights[i][j]+(lr*hidscore[j]*ErrVecOutD[i])

    	ErrVecInD = util.Counter()
    	for j in range(HID_LAY):
    		ErrVecInD[j]= (1/(1+pow(math.e,(trainingData[j]*self.inweights[j]))))*(1-(1/(1+pow(math.e,(trainingData[j]*self.inweights[j])))))

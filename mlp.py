# mlp.py
# -------------

# mlp implementation
import util
import math
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
	self.HID_LAY = 10

	for x in range(self.HID_LAY):
		self.hidweights[x] = util.Counter()

	for label in legalLabels:
		self.inweights[label] = util.Counter()


	  
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):

	self.features = trainingData[0].keys()

	import random

	for j in range(self.HID_LAY):
		for f in self.features:
			self.inweights[j][f] = random.randint(1,6)

	for j in range(len(self.legalLabels)):
		for f in range(self.HID_LAY):
			self.hidweights[j][f] = random.randint(1,6)

	for iteration in range(self.max_iterations):
		print "Starting iteration ", iteration, "..."
		correct = 0;
		incorrect = 0;

		for i in range(len(trainingData)):
			hidscore = util.Counter()
			fscore = util.Counter()

			for j in range(len(self.hidweights)):
				hidscore[j] = trainingData[i]*self.inweights[j]
				print hidscore

			for l in range(len(self.legalLabels)):
				#print hidscore
				#print self.hidweights[l]
				#print hidscore*self.hidweights[l]

				fscore[l] = 1.0/(1.0+pow(math.e,-1.0*(hidscore*self.hidweights[l])))
				#print fscore

			prediction = fscore.argMax()

			if prediction != trainingLabels[i]:
				ErrVecOut = util.Counter()
				ErrVecOutD = util.Counter()

				for k in range(len(self.legalLabels)):
					ErrVecOut[k]=(trainingLabels[k]-prediction)
					ErrVecOutD[k]= ErrVecOut[k]*(1.0/(1.0+pow(math.e,-1.0*(hidscore*self.hidweights[k]))))*(1.0-(1.0/(1.0+pow(math.e,-1.0*(hidscore*self.hidweights[k])))))

				#MLPClassifier.backProp(self,0.30,hidscore,trainingData,ErrVecOut,ErrVecOutD)



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
			for j in range(self.HID_LAY):
				self.hidweights[i][j]= self.hidweights[i][j]+(lr*hidscore[j]*ErrVecOutD[i])

		ErrVecInD = util.Counter()
		for j in range(self.HID_LAY):
			ErrVecInD[j]= (1.0/(1.0+pow(math.e,-1.0*(trainingData[j]*self.inweights[j]))))*(1.0-(1.0/(1.0+pow(math.e,-1.0*(trainingData[j]*self.inweights[j])))))

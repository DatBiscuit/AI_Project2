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
			f1score = util.Counter()


			for j in range(len(self.hidweights)):
				#hidscore[j] = 10.0/(1.0+pow(math.e,-1.0*(trainingData[i]*self.inweights[j])))
				#hidscore[j] = math.tanh(trainingData[i]*self.inweights[j])
				hidscore[j]=math.tanh((trainingData[i]*self.inweights[j]/10000.0))
			#print hidscore

			for l in range(len(self.legalLabels)):
				#print hidscore
				#print self.hidweights[l]
				#print hidscore*self.hidweights[l]

				f1score[l] = math.tanh((hidscore*self.hidweights[l]/10000.0))
				#fscore[l] = (hidscore*self.hidweights[l])
				
			#print fscore
			#print
			

			#fscore2 = util.normalize(fscore)
			#print fscore2
			#print
			#print f1score
			#print
			f1score2 = util.normalize(f1score)
			#print f1score2
			#print
			
			prediction = f1score2.argMax()

			if prediction != trainingLabels[i]:
				ErrVecOut = util.Counter()
				ErrVecOutD = util.Counter()
				for j in range(len(self.legalLabels)):
					ErrVecOut[j]=(1.0-f1score2[j])
					ErrVecOutD[j]= ErrVecOut[j]*(1-pow(f1score2[j],2))

				MLPClassifier.backProp(self,10.0,hidscore,trainingData,ErrVecOut,ErrVecOutD)
				incorrect+=1
			else:
				correct+=1

			print(str(correct),str(incorrect))



	#util.raiseNotDefined()
	
  def classify(self, data ):
	guesses = []
	for datum in data:
		midvectors = util.Counter()
		for l in range(self.HID_LAY):
			midvectors[l] = self.inweights[l]*datum
		fvec = util.Counter()
		for i in self.legalLabels:
			fvec[i] = self.hidweights[i]*midvectors
		guesses.append(fvec.argMax())


	  # fill predictions in the guesses list


	  
	  #util.raiseNotDefined()
	return guesses

  def backProp(self,lr,hidscore,trainingData,ErrVecOut,ErrVecOutD):
		for i in range(len(self.legalLabels)):
			for j in range(self.HID_LAY):
				self.hidweights[i][j]= self.hidweights[i][j]+(lr*hidscore[j]*ErrVecOutD[i])

		ErrVecInD = util.Counter()
		for j in range(self.HID_LAY):
			ErrVecInD[j]= (1-pow(hidscore[j],2))*(self.inweights[j]*ErrVecOutD)


		for i in range(self.HID_LAY):
			for j in range(len(self.features)):
				self.inweights[i][j]= self.inweights[i][j]+lr+trainingData[i][j]+ErrVecInD[i]



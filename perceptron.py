# perceptron.py
# -------------

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING


    # Randomizing Weights
    import random
    for j in range(len(self.legalLabels)):
      for f in self.features:
        
        self.weights[j][f] = random.randint(1,2)
        #print self.weights[j][f]

    #self.setWeights([A]*len(self.legalLabels))
    #self.weights = [A]*len(self.legalLabels)


    
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      #print "total iterations", self.max_iterations

      correct = 0;
      incorrect = 0;

      for i in range(len(trainingData)):

        # Score calculation
        score = util.Counter()
        for j in range(len(self.legalLabels)):
          score[j] = trainingData[i] * self.weights[j]
        prediction = score.argMax()

        #print(score)
        #print("Prediction:", str(prediction), "Actual:", str(trainingLabels[i]))

        # Update Step
        if prediction != trainingLabels[i]:
          #if condition about error function
          self.weights[prediction] -= trainingData[i]
          self.weights[trainingLabels[i]] += trainingData[i]
          incorrect -= 1
        else:
          correct += 1

      # Metrics
      print (str(correct), str(incorrect))





      #util.raiseNotDefined()
    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    #finding the largest weight values in matrix x

    "*** YOUR CODE HERE ***"


    util.raiseNotDefined()

    return featuresWeights


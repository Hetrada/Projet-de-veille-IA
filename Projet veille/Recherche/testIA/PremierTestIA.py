import numpy as np 

x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float) # Donnée de sortie 1->Rouge et 0->Bleu

parameters = open('Recherche\\testIA\\IAParamaters.txt')
oneParameters = False

x_entrer = x_entrer/np.amax(x_entrer, axis=0)

X = np.split(x_entrer, [8])[0]
xPrediction = np.split(x_entrer,[8])[1]


class Neural_Network(object):
	"""docstring for Neural_Network"""
	def __init__(self, text, count):
		self.inputSize = 2
		self.outputSize = 1
		self.hidenSize = 3
  
		self.W1 = np.random.randn(self.inputSize, self.hidenSize) # Matrice 2x3
		self.W2 = np.random.randn(self.hidenSize, self.outputSize) # Matrice 3x1

		if text.read() != "":
			text.close()
			with open('Recherche\\testIA\\IAParamaters.txt') as text:
				count = True
				self.w1 = text.readline()[2]
				text.close()
				with open('Recherche\\testIA\\IAParamaters.txt') as text:
					self.w2 = text.readline()[3]


		

	def forward(self, X):

		self.z = np.dot(X,self.W1)
		self.z2 = self.sigmoid(self.z)
		self.z3 = np.dot(self.z2,self.W2)
		o = self.sigmoid(self.z3)
		return o
	def sigmoid(self,s):
		return 1/(1+np.exp(-s))

	def sigmoidPrime(self,s):

		return s*(1-s)

	def backward(self,X,y,o):


		self.o_error = y-o
		self.o_delta = self.o_error * self.sigmoidPrime(o)

		self.z2_error = self.o_delta.dot(self.W2.T)
		self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

		self.W1 += X.T.dot(self.z2_delta)
		self.W2 += self.z2.T.dot(self.o_delta)

	def train(self,X,y):

		out = self.forward(X)
		self.backward(X,y,out)

	def predict(self):
		print("Donnée prédite après entraînement: ")
		print("Entrée: \n" + str(xPrediction))
		print("Sortie: \n" + str(self.forward(xPrediction)))

		if self.forward(xPrediction) < 0.5:
			print("La fleur est BLEU !\n")
		else:
			print("La fleur est ROUGE !\n")


NN = Neural_Network(parameters, oneParameters)

for i in range(30000):
	print("# "+str(i)+"\n")
	print("Valeurs d'entrées: \n" + str(X))
	print("Valeur actuelle: \n" + str(y))
	print("Sortie Prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
	print("\n")
	NN.train(X,y)

NN.predict()
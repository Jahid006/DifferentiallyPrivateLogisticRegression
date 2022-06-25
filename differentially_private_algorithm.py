from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn
import tqdm
import numpy as np

class Logistic_Regression():
        
    def __init__(self, X, y, 
                alpha=.1,
                iteration = 100 ,
                epsilon = 0.01, 
                delta = 1e-5,
                differentially_private = False,
                regularization = 0,
                test_split = 0.125,
                random_state = 41
    ):


        self.x_train, self.x_val, self.y_train, self.y_val  =  train_test_split(X, y, test_size = test_split, random_state = random_state)
    
        self.n_iteration = iteration
        self.alpha = alpha
        
        self.epsilon = epsilon
        self.differentially_private = differentially_private
        self.delta = delta
        self.regularization =  regularization

        self.m_train , self.n_train = self.x_train.shape
        self.x_train = np.append(np.ones((self.m_train,1)), self.x_train, axis=1)
        self.y_train = self.y_train.reshape(self.m_train,1)

        self.initial_theta = np.random.randn(self.n_train+1,1)*.01
        self.train_history = []
        self.cv_history = []
        self.theta_history =  []
        
        self.m_val , self.n_val = self.x_val.shape[0], self.x_val.shape[1]
        self.x_val = np.append(np.ones((self.m_val,1)), self.x_val, axis=1)
        self.y_val = self.y_val.reshape(self.m_val,1)       
    
    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))-0.000001
    
    def costFunction(self, theta, X, y):
        m = len(y)
        predictions = self.sigmoid(np.dot(X,theta))
        
        if self.differentially_private:  # Q.4 if Enabled
            b = (self.delta/self.epsilon)  #laplacian scaling
            z = np.array( [ np.random.laplace(0,b) for i in range(X.shape[0])]).reshape(-1,1)  #Noise
            predictions = predictions + z
        
        cost = sklearn.metrics.log_loss(y,predictions)
        grad = (1/m) * np.dot(X.transpose(),(predictions - y))

        if self.regularization:
            regularization_cost = np.sum(np.dot(theta.T,theta))/(2*m)
            regularization_cost = (1/m)* self.regularization * regularization_cost
            cost += regularization_cost
            grad +=  self.regularization*theta

        return cost , grad

    
    def gradientDescent(self, verbose):

        theta = self.initial_theta
        train_history =[]
        cv_history =[]
        theta_history = []
        
        for i in tqdm.tqdm(range(self.n_iteration)):
            cost, grad = self.costFunction(theta, self.x_train, self.y_train)
            theta = theta - (self.alpha * grad)

            cv_cost, _ = self.costFunction(theta, self.x_val, self.y_val)
            if verbose: print("Iteration: {}, Train-Loss:  {}, CV-Loss:  {}".format(i+1,  cost, cv_cost))

            train_history.append(cost)
            cv_history.append(cv_cost)
            theta_history.append(theta)

        print("Stats: Iterations {}, Train-Loss:  {}, CV-Loss:  {}".format(i+1,  cost, cv_cost))

        return theta , train_history, cv_history, theta_history
    
    def AlphavsIteration(self, loss_threshold = .5):
        
        theta = self.initial_theta
        n_iterations = 0

        while True:
            n_iterations += 1
            cost, grad = self.costFunction(theta, self.x_train, self.y_train)
            theta = theta - (self.alpha * grad)
            
            if cost <= loss_threshold: return n_iterations

            cv_cost, _ = self.costFunction(theta, self.x_val, self.y_val)

            if self.verbose: print("Iteration: {}, Train-Loss:  {}, CV-Loss:  {}".format(n_iterations+1,  cost, cv_cost))

    def fit(self, verbose = False):
        theta , train_history, cv_history, theta_history = self.gradientDescent(verbose)
        self.accumulate_artifacts(theta , train_history, cv_history, theta_history)

        return theta , train_history, cv_history, theta_history
    
    def fit_sample(self):
        theta , train_history, cv_history, theta_history = self.gradientDescent()
        return train_history[-1], cv_history[-1]

    def predict(self, X, Y,threshold =0.5):  
        X = np.append(np.ones((len(X),1)), X, axis=1)
        probabilities = 1/(1 + np.exp(-np.dot(X,self.theta)))
        prediction = np.ones(len(probabilities))
        prediction[probabilities.reshape(-1)<threshold] = 0
        
        return sum((Y*1.0 == prediction))/len(Y), probabilities, prediction    

    def accumulate_artifacts(self ,theta,  train_history, cv_history, theta_history):
        self.theta = theta
        self.train_history.extend(train_history)
        self.cv_history.extend(cv_history)
        self.theta_history.extend(theta_history)

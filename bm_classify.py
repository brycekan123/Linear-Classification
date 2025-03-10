import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        for i in range(max_iterations):
            #initialize gradientw and gradientb for new iteration
            gradw = np.zeros_like(w)
            gradb = 0
            for feat_index in range(len(X)):
                #because our labels are {0,1}, y'i = 2yi-1. 
                #this converts labels to {1,-1}, which allows us to classifiy mismatch
                y_prime = 2*y[feat_index]-1
                #prediction equation
                predicted_y = np.dot(w.T, X[feat_index])+b
                #aligned prediction with -1 and 1
                if y_prime * predicted_y <= 0: #misclassified point. update gradient and b
                    #get sum of gradient
                    gradw += -1*y_prime * X[feat_index] #gradient w of prediction
                    gradb += -1*y_prime #gradient b of prediction
            #update weights and b for entire dataset  
            gradw /= len(X)
            gradb /= len(X)
            w = w - step_size * gradw
            b = b - step_size * gradb
    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for i in range(max_iterations):
            #initialize gradients to 0
            gradw_L = np.zeros_like(w) 
            gradb_L = 0
            for feat_index_2 in range(len(X)):
                #get predicted point
                predicted = (np.dot(w.T, X[feat_index_2])+b)
                sgpred = sigmoid(predicted)
                #logistic loss function
                loss = -(y[feat_index_2]*np.log(sgpred)+(1-y[feat_index_2])*(np.log(1-sgpred)))
                #if misclassified(loss!=0), get the summation of gradient w.r.t w and b
                if loss != 0:
                    gradw_L += (sgpred - y[feat_index_2])*X[feat_index_2]
                    gradb_L += (sgpred - y[feat_index_2])
            #take the avereage of the sum of gradient for that round
            gradw_L = gradw_L/len(X)
            gradb_L = gradb_L/len(X)
            w = w - step_size* gradw_L
            b = b - step_size * gradb_L
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1/(1+np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    preds = np.zeros(N, dtype=int)
    
    for i in range(len(X)):
        #convert predicts back to 0 and 1 labels
        predict = np.dot(w.T, X[i])+b
        if predict < 0:
            preds[i] = 0
        else:
            preds[i] = 1
    
    assert preds.shape == (N,) 
    return np.array(preds)


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":
        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            #weight: Shape(3,2), X[n]: 1400x2 (1126th row)
            x = X[n].reshape(1, -1)  # Shape (1,) -> (1,2) so can multiply with w
            z = np.dot(w,x.T)#3x2 x 2x1 = 3,1. z = raw predicted score(before bias)
            C = w.shape[0]
            #adds bias term across entire Z row based on class. b = (3,1) or Cx1
            for c in range(C):
                z[c, :] += b[c]
            exp_z = np.exp(z - np.max(z,axis=0)) # for numerical stability in softmax function. Shape: (3x1)
            softmax_z = exp_z / np.sum(exp_z,axis=0) #exponent /summation all datapoints to the exp. Shape: (3x1)
            #apply one-hot representation where correct class' index is 1 and all others are 0. 
            true_label = np.zeros((C,1)) # Shape
            true_label[y[n]] = 1
            #softmax_z - true label will give positive number if incorrectly classified(probability - 0),
            # but subtracting 1 will result in negative gradient, 
            # increasing weight of correct class. Vice versa with incorrect class 
            gradW = np.dot((softmax_z-true_label),x) #shape: (3,2)
            gradB = (softmax_z-true_label).reshape(-1) #shape: (3,)
            w = w - step_size * gradW
            b = b - step_size * gradB


    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        
        for it in range(max_iterations):
            #Usually, I would have a nested for loop to iteratively add the 
            #gradients and update the weights all at once with gradient descent.
            #Because there are so many data points, I used matrix multiplication 
            #to efficiently compute predicted raw scores
            z = np.dot(w,X.T) #Shape: 3x1400
            #apply all biases
            for c in range(C):
                z[c, :] += b[c]
            # Initialize gradient for all weights
            gradW = np.zeros(w.shape)
            gradB = np.zeros(C)  
            exp_z = np.exp(z - np.max(z, axis=0)) #Shape: (3x1400)
            softmax_z = exp_z / np.sum(exp_z, axis=0)  #Shape: (3x1400)
            true_label = np.zeros((C,N))
            #Sets 1 for true class index for every data point. rest are set to 0
            true_label[y, np.arange(N)] = 1  
            #get difference, and update weights vased on gradient
            gradW = np.dot(softmax_z - true_label, X)  # Shape: (C, D)
            gradB = np.sum(softmax_z - true_label)  # Shape: (C,)
            w = w - step_size * gradW/N # average 
            b = b - step_size * gradB/N
    else:
        raise "Undefined algorithm."
    
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    #gets number of classes
    C = w.shape[0]
    z = np.dot(w,X.T) #CxD, 3x1400
    #adds bias term to all points in that class
    for c in range(C):
        z[c, :] += b[c]
    exp_z = np.exp(z - np.max(z))
    softmax_z = exp_z / np.sum(exp_z) #Shape: 3,1400 
    #softmax_z is a 3x1400 vector of all probabilities
    #Performs columnwise argmax and returns the index of the class with 
    #the highest probability for each datapoitn
    preds = np.argmax(softmax_z,axis=0) # Shape: (1400,)
    assert preds.shape == (N,)
    return preds




        
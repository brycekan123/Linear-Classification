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
        #perceptron loss equation: yi*(wt*xi+b)
        for i in range(1):
            #initialize gradientw and gradientb for new iteration
            gradw = np.zeros_like(w)
            gradb = 0
            misclassified = 0
            for feat_index in range(len(X)):
                #because our labels are {0,1}, y'i = 2yi-1. 
                #this converts labels to {1,-1}, which allows us to classifiy mismatch
                y_prime = 2*y[feat_index]-1
                #prediction equation
                #gives a raw prediction. can be any negative, positive or 0 number. 
                predicted_y = np.dot(w.T, X[feat_index])+b
                #based on the sign, I know if its misclassified.
                if y_prime * predicted_y <= 0: #misclassified point. update gradient and b
                    misclassified +=1
                    #get sum of gradient
                    gradw += -1*y_prime * X[feat_index] #gradient w of prediction
                    gradb += -1*y_prime #gradient b of prediction
                
            #update weights and b for entire dataset  
            #getting average of gradients
            gradw = gradw / len(X)
            gradb = gradb / len(X)
            w = w - step_size * gradw
            b = b - step_size * gradb
            
        print("Done with Perceptron Loss")
    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################        
        print("in logistic loss")
        for i in range(1):
            #initialize gradients
            gradw_L = np.zeros_like(w) 
            gradb_L = 0
            num_mis = 0
            for feat_index_2 in range(len(X)):
                predicted = (np.dot(w.T, X[feat_index_2])+b)
                #apply sigmoid to predicted labels
                sgpred = sigmoid(predicted)
                #logistic loss function equation
                loss = -(y[feat_index_2]*np.log(sgpred)+(1-y[feat_index_2])*(np.log(1-sgpred)))
                # if loss is not 0, adjust weights and biases
                #if there is large difference between sigmoid predicted and actual, the gradient change will be large
                if loss != 0:
                    num_mis +=1
                    gradw_L += (sgpred - y[feat_index_2])*X[feat_index_2]
                    gradb_L += (sgpred - y[feat_index_2])

            #take the avereage of the sum of gradient for that round. only dividing misclassified points
            gradw_L = gradw_L/len(X)
            gradb_L = gradb_L/len(X)
            w = w - step_size* gradw_L
            b = b - step_size * gradb_L
            #print(w,b)
        print("Done with Logistic Loss")
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
    #initialize empty array
    preds = np.zeros(N, dtype=int)
    for i in range(len(X)):
        #standard prediction equation
        predict = np.dot(w.T, X[i])+b
        #since our labels are {0,1}, i adjust {-1,1} -> {0,1}
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
    print(C,"Classes")
    print("this is y", y)
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":
        for it in range(40):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #


            ####################################################	
            #vector = X[n]
            #z = predicted - max(predicted values in that vector)
            #softmax = exp**z/(sum of all z in that vector)
            predicted = np.dot(w, X[n])+b
            print("actual value:", y[n])
            ##transpose is not present in predicted
            
            #ALL PREDICTED ARE THE SAME
            #SOFTMAX function works fine!
            z = predicted - np.max(predicted)
            print("Prediction!",predicted)
            print(np.max(predicted))
            print(z)
            expz = np.exp(z)
            softmax= expz/(np.sum(expz))
            #this is predicted values but softmaxed. now, i need to do log on each one?
            print("softmax",softmax)
            print(softmax.shape)#3x1
            # n is my random index across the sample data so i don't need loop. I use this to replaceX[feature_index)]
            #also use this for my index for y. 
            gradw = np.array([])
            gradb = np.array([])
            for softZ in softmax:
                #i don't think im subtracting y[n]. that is just the label 0 and 1.
                loss = -(y[n]*np.log(softZ)+(1-y[n])*(np.log(1-softZ)))
                #print("LOSS", loss)
                #THIS MEANS point is misclassified!
                print(loss,"LOSS")
                if loss != 0:
                    #this is somehow not a singular value. X[n] is 2x1 array.
                    gradw_L = (softZ - y[n])*X[n]
                    print(gradw_L,"gradw_L")
                    #add w to array
                    #print("gradw_L", gradw_L)
                    gradw = np.append(gradw,gradw_L)
                    gradb_L = (softZ - y[n])
                    gradb = np.append(gradb,gradb_L)
                    print(gradw,"GRADW")
                    #calculate gradient IMMEDIATELY and then update weights and biases
                    w = w - step_size* gradw
                    b = b - step_size * gradb
                    print("weights",w,"b",b)
            # print("weights!", w)
            # print(w.shape)


            # #logistic loss function equation
            # loss = -(y[n]*np.log(predicted)+(1-y[n])*(np.log(1-predicted)))
            # if loss is not 0, adjust weights and biases
            #if there is large difference between sigmoid predicted and actual, the gradient change will be large
            # if loss != 0:
            #     num_mis +=1
            #     gradw_L = (predicted - y[n])*X[n]
            #     gradb_L = (predicted - y[n])
            #     #calculate gradient IMMEDIATELY and then update weights and biases
            #     w = w - step_size* gradw_L
            #     b = b - step_size * gradb_L

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        
        

    #else:
        print("gd")
        #raise "Undefined algorithm."
    

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

    
    assert preds.shape == (N,)
    return preds




        
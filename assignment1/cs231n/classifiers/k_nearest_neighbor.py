import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]  # 현재시점에서는 500개가 들어올 것임
    num_train = self.X_train.shape[0]  # 현재시점에서는 5000개가 들어와 있음
    dists = np.zeros((num_test, num_train))  # (500, 5000) matrix 일 것임. 즉 각각의 test 이미지(=행)에 대한 각각의 train 이미지(=열)의 거리값이 들어간다
    
    for i in range(num_test):  # python 3에서는 xrange --> range
      for j in range(num_train):  # python 3에서는 xrange --> range
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #pass
        # L2 (Euclidean) distance 두 matrix의 차이의 제곱을 모두 더하는 것 - 그다음 로그를 취하는 것은 생략가능
        #dists[i,j] = np.sum(np.power(np.subtract(X[i], self.X_train[j])))
        temp = np.power(X[i]- self.X_train[j], 2)
        #dists[i,j] = np.log(np.sum(temp))
        dists[i,j] = np.sum(temp)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):  # python 3에서는 xrange --> range
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.
    - 테스트이미지와 학습이미지간의 거리정보
    
    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i]. 
    - 예측한 값 (아마도 0에서 9 사이) 리턴
    """
    num_test = dists.shape[0]  # 테스트 이미지 개수
    y_pred = np.zeros(num_test)  # 테스트 이미지 하나하나에 대한 예측값을 저장할 공간
    
    for i in range(num_test): # python 3에서는 xrange --> range
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      # 테스트 i번째 이미지에 대해 가장 가까운 k개의 이미지의 y값(=라벨)
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # 헷갈려서 헤메다가 정답을 컨닝: https://goo.gl/UvSLng
      order = np.argsort(dists[i])  
      #  argsort 테스트 이미지의 거리값들이 작은 순서대로 인덱스 값을 리턴해준다
      # 예를 들어 np.argsort([3,5,2,6]) 은 [(가장작은 2의 인덱스인)2, (3의 인덱스인) 0, (각각 5,6의 인덱스인) 1,3] 즉, [2,0,1,3]
                
      closest_y = self.y_train[order]  # 이렇게 하면 order에 들어가있는 인덱스대로 총 5000개가 closest_y에 들어가게 된다. 
      closest_y = closest_y[:k]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      # closest_y 중에서 가장 많은 값 나온 라벨을 찾기. 동률이면 앞의 라벨    #
      #########################################################################
      # https://goo.gl/RqMsdM
      # print(closest_y)
      counts = np.bincount(closest_y) 
      print(counts)
      y_pred[i] = np.argmax(counts)
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred


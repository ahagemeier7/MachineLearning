import numpy as np

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self, x, y):
    self.x_train = x
    self.y_train = y


  def predict(self, x):
    predictions = [self._predict(xs) for xs in x]

  def _predict(self, x):
    #compute distances 
    distances = [euclidean_distance(x,x_train) for x_train in self.x_train]

    #get the closest k 
    np.argsort(distances)[:self.k]

    #majority vote
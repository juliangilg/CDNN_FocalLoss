from sklearn.base import BaseEstimator,ClassifierMixin
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

class DNN_FocalLoss(BaseEstimator,ClassifierMixin):
  def __init__(self,loss=None):
    self.loss = loss

  def model(self, P):
    # Define the input layer
    inputs = Input(shape=(P,))

    # Define the hidden layers
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Define the output layer
    outputs = Dense(2, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model
  
  def fit(self, X, y):
    N, P = X.shape
    self.model_ = self.model(P)
    self.model_.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
    self.model_.fit(X, y, epochs=100, batch_size=64)
    return self

  def predict(self, X):
    y_est = self.model_.predict(X)
    return y_est

class MA_DNN_FocalLoss(BaseEstimator,ClassifierMixin):
  def __init__(self,loss, K, R):
    self.loss = loss
    self.K = K
    self.R = R

  def model(self, P):
    # Define the input layer
    inputs = Input(shape=(P,))

    # Define the hidden layers
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Define ground truth layer
    zeta = Dense(self.K, activation='softmax')(x)

    # Define reliability layer
    lam = Dense(self.R, activation='sigmoid')(x)

    # Define the output layer
    out = tf.keras.layers.Concatenate()([zeta, lam])

    # Create the model
    model = Model(inputs=inputs, outputs=out)
    return model
  
  def fit(self, X, y):
    N, P = X.shape
    self.model_ = self.model(P)
    self.model_.compile(optimizer='adam', loss=self.loss)
    self.model_.fit(X, y, epochs=100, batch_size=100)
    return self

  def predict(self, X):
    y_est = self.model_.predict(X)
    return y_est

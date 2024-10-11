import tensorflow as tf

class CategoricalFocalLoss(tf.keras.losses.Loss):
  def __init__(self, gamma=2.0):
    super().__init__()
    self.gamma = gamma

  def call(self, y_true, y_pred):
    """
    Categorical focal loss function.

    Args:
      y_true: True labels (integers).
      y_pred: Predicted probabilities (output of softmax).
      gamma: Focal loss parameter.

    Returns:
      Categorical cross-entropy loss.
    """
    epsilon=1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) # Preventing log(0)
    w = (1 -  y_pred)**self.gamma  # alpha*(1 - p(k|x))^gamma
    num = -tf.gather(w*tf.math.log(y_pred), tf.cast(y_true, tf.int32), axis=-1, batch_dims=1)
    # den = -tf.reduce_sum(w*tf.math.log(y_pred), axis=1, keepdims=True)
    den = 1
    loss = tf.reduce_mean(num / den)
    return loss


class MA_CategoricalFocalLoss(tf.keras.losses.Loss):
  def __init__(self, K, R,gamma=2):
    super().__init__()
    self.K = K
    self.R = R
    self.gamma = gamma


  def call(self, y_true, y_pred):
    ####### extract zeta and lambda
    zeta = tf.cast(y_pred[:,:self.K], dtype=tf.float32)
    lam = tf.cast(y_pred[:,self.K:], dtype=tf.float32)
    y_true = tf.cast(y_true, tf.int32)


    epsilon=1e-7
    zeta = tf.clip_by_value(zeta, epsilon, 1. - epsilon) # Preventing log(0)

    ##### normalized cross-entropy term, lambda = 1
    yhat = tf.cast(tf.one_hot(y_true, self.K, axis=1), dtype=tf.float32)
    aux = tf.math.multiply((1-zeta)**self.gamma, tf.math.log(zeta))

    y_pred_ = tf.repeat(tf.expand_dims(aux, axis=2), self.R, axis=2)

    log_zeta = tf.reduce_sum(tf.math.multiply(y_pred_,yhat), axis=1)
    num = tf.reduce_sum(tf.math.multiply(lam,log_zeta), axis = 1)

    #den = tf.reduce_sum(aux, axis=1, keepdims=True)
    den = -1
    first_term = num / den

    ## Random term lambda=0
    second_term = tf.reduce_sum(tf.math.log(1/self.K)*(1 - lam), axis=1)
    loss = tf.reduce_mean(first_term - second_term)
    return loss

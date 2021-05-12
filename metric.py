import tensorflow as tf

class MultilabelAccuracy(tf.keras.metrics.Metric):
  """
  Class to calculate hamming score metric for the model
  """
  
  def __init__(self, threshold = 0.5, name = 'multi_label_accuracy', **kwargs):
    """
    constructor class for the metric function
    inputs:
        - threshold : float
              threshold to classify the probabilities, should be between 0 and 1
        - name : str
              name of the metric to be used while fitting the model
    """
    super(MultilabelAccuracy, self).__init__(name = name, **kwargs)
    self.threshold = threshold
    self.total = self.add_weight(name = 'total', initializer = 'zeros', dtype = tf.float64)
    self.count = self.add_weight(name = 'count', initializer = 'zeros', dtype = tf.float64)

  def update_state(self, y_true, y_pred, sample_weight = None):
    """
    updates the total and count variable calculated for each batch
    inputs:
        - y_true : tensor
              true probabilities of the samples in batch
        - y_pred : tensor
              predicted probabilities of the samples in batch
    """
    y_pred = tf.cast(y_pred, dtype = tf.float32)
    y_pred = tf.greater(y_pred, self.threshold)
    y_true = tf.cast(y_true, tf.bool)

    and_op = tf.math.logical_and(y_true, y_pred)
    or_op = tf.math.logical_or(y_true, y_pred)

    count = tf.cast(tf.math.count_nonzero(or_op, axis = 1),tf.float32)
    epsilon_added = tf.py_function(self.get_zero_count, inp = [count], Tout = tf.float32)
  
    values = tf.divide(tf.cast(tf.math.count_nonzero(and_op, axis = 1),tf.float64),tf.cast(epsilon_added,tf.float64))
    sample_size = tf.shape(y_true)[0]
    
    self.total.assign_add(tf.reduce_sum(values,0))
    self.count.assign_add(tf.cast(sample_size, tf.float64))

  def result(self):
    """
    returns total accuracy calculate so far
    returns:
        - accuracy : float
              total accuracy for all the batches seen so far
    """
    return self.total/self.count

  def reset_states(self):
    """
    values to reset to after each epoch
    """
    self.total.assign(0.0)
    self.count.assign(0.0)

  def get_zero_count(self,count):
    """
    utility function
    """
    indices = tf.where(count == 0)
    sparse_mask = tf.SparseTensor( values=[1e-7]*len(indices), indices=indices, dense_shape=list(tf.shape(count)))
    return tf.math.add(count,tf.cast(tf.sparse.to_dense(sparse_mask),dtype = tf.float32))
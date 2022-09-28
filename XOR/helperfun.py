import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
import itertools
import subprocess

def run_bash(cmd, myoutput):
    p = subprocess.Popen(cmd, shell=True, stdout=myoutput, stderr=myoutput, executable='/bin/bash')
    # out = p.stdout.read().strip()
    # return out  # This is the stdout from the shell command

def get_dict_by_key(dict, keysNames):
  return {k: dict[k] for k in dict.keys() & keysNames}

def build_string_from_dict(d, sep='%'):
    """
     Builds a string from a dictionary.
     Mainly used for formatting hyper-params to file names.
     Key-value pairs are sorted by the key name.

    Args:
        d: dictionary

    Returns: string
    :param d: input dictionary
    :param sep: key-value separator

    """

    return sep.join(['{}={}'.format(k, _value2str(d[k])) for k in sorted(d.keys())])


def _value2str(val):
    if isinstance(val, float): 
        # %g means: "Floating point format.
        # Uses lowercase exponential format if exponent is less than -4 or not less than precision,
        # decimal format otherwise."
        val = '%g' % val
    else:
        val = '{}'.format(val)
    val = re.sub('\.', '_', val)
    return val

def plot_weights(weights):
  """
  Returns a matplotlib figure containing the plotted weight matrix.

  Args:
    weights (matrix, shape = [n, m]): a weight matrix
  """
  if weights.shape[0] > weights.shape[1]:
    if len(weights.shape) == 3:
      weights = tf.transpose(weights, perm=[1,0,2])
    else:
      weights = tf.transpose(weights, perm=[1,0])

  figure = plt.figure(figsize=(weights.shape[1], weights.shape[0]+.5))
  plt.imshow(weights, cmap=plt.cm.viridis)
  plt.title("Weight matrix")
  plt.colorbar(orientation='horizontal', pad=0.3)
  xTickMs = np.array(range(0, weights.shape[1]))
  plt.xticks(xTickMs, [str(x) for x in xTickMs+1], rotation=45)
  yTickMs = np.array(range(0, weights.shape[0]))
  plt.yticks(yTickMs, [str(x) for x in yTickMs+1])

  # Use white text if squares are dark; otherwise black.
  threshold = tf.reduce_max(weights) / 2.
  for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
    color = "white" if weights[i, j] < threshold else "black"
    plt.text(j, i, '%2.2f' % weights[i, j], horizontalalignment="center", color=color)

#   plt.tight_layout()
  return figure

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
import matplotlib.pyplot as plt
import numpy as np


def visulaize_number(data, label):

    plt.figure(figsize=(20,4))
    for index, (image, label) in enumerate(zip(data,label)):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % label, fontsize = 20)
import torch
import torchvision
import numpy as np
import time
import os
import matplotlib
# Add this to save all the plot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LSTM_epo = 50
ATTE_epo = 50

with open('lstm_{}.txt'.format(LSTM_epo)) as loss:
    # Danger but I don't care.
    loss.readline() # Throw away training loss
    lstm_loss = eval(loss.readline())
with open('attention_{}.txt'.format(ATTE_epo)) as loss:
    # Danger but I don't care.
    loss.readline() # Throw away training loss
    attention_loss = eval(loss.readline())

plt.clf()
plt.plot(lstm_loss, label="LSTM Validation loss")
plt.plot(attention_loss, label="Attention Validation loss")
plt.legend(loc="upper left")
plt.savefig("3model.png")
print("Plot Finished.")

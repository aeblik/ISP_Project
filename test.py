from utils import plot_frequency_spectrum
import numpy as np
# Create a dummy data array
data = np.random.rand(1024)
samplerate = 44100
plot_frequency_spectrum(data, samplerate, "title")
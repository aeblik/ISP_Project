import numpy as np
from scipy.fft import fft
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def plot_frequency_spectrum(data, samplerate, title="Frequency Spectrum"):
    """Plot the frequency spectrum of mono or stereo audio data on the same row and make plots smaller."""
    # Check if the data is mono or stereo and set the number of channels accordingly
    channels = data.shape[1] if data.ndim > 1 else 1
    
    # Set the figure size based on the number of channels
    fig_width = 6 * channels  # Smaller and proportional to the number of channels
    plt.figure(figsize=(fig_width, 3))  # Reduced height to make the plot more compact

    # Process each channel
    for i in range(channels):
        ax = plt.subplot(1, channels, i + 1)  # Ensure all plots are on the same row
        channel_data = data[:, i] if channels > 1 else data

        # Perform the FFT
        fft_result = fft(channel_data)
        magnitude = np.abs(fft_result)
        frequency = np.linspace(0, samplerate / 2, len(magnitude)//2)

        # Plot the spectrum
        ax.plot(frequency, magnitude[:len(magnitude)//2])
        ax.set_title(f"{title} - Channel {i+1}")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([0, samplerate / 2])
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_time_domain(data, samplerate, title="Audio Signal"):
    """Plot audio data that can be either mono or stereo, arranging plots side by side for stereo."""
    duration = len(data) / samplerate
    times = np.linspace(0, duration, num=len(data))
    
    channels = data.shape[1] if data.ndim > 1 else 1

    # Set the figure size based on the number of channels
    fig_width = 6 * channels  # Scaled width to accommodate all plots in one row
    fig_height = 3  # Reduced height to make the plots compact

    plt.figure(figsize=(fig_width, fig_height))

    for i in range(channels):
        ax = plt.subplot(1, channels, i + 1)  # Ensure all plots are on the same row
        channel_data = data[:, i] if channels > 1 else data

        # Plot the time domain data
        ax.plot(times, channel_data, label=f'Channel {i+1}', color='b' if i == 0 else 'r')
        ax.set_title(f"{title} - Channel {i+1}")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_frequency_spectrum_2(data, samplerate, title="Frequency Spectrum"):
    """Plot the frequency spectrum of mono or stereo audio data on the same row and make plots smaller."""
    # Check if the data is mono or stereo and set the number of channels accordingly
    channels = data.shape[1] if data.ndim > 1 else 1
    
    # Set the figure size based on the number of channels
    fig_width = 6 * channels  # Smaller and proportional to the number of channels
    plt.figure(figsize=(fig_width, 3))  # Reduced height to make the plot more compact
    N = len(data[0])

    # Process each channel
    for i in range(channels):
        ax = plt.subplot(1, channels, i + 1)  # Ensure all plots are on the same row
        channel_data = data[:, i] if channels > 1 else data

        # Perform the FFT
        fft_result = fft(channel_data)
        magnitude = 1/N*np.abs(fft_result)
        frequency = np.linspace(0, samplerate / 2, len(magnitude)//2)

        # Plot the spectrum
        ax.plot(frequency, magnitude[:len(magnitude)//2])
        ax.set_title(f"{title} - Channel {i+1}")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([0, samplerate / 2])
        ax.grid(True)

    plt.tight_layout()
    plt.show()
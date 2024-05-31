import numpy as np
import soundfile as sf
import scipy.signal
import sounddevice as sd

class AudioEffects:
    def __init__(self, filename):
        self.original_data, self.samplerate = sf.read(filename)
        # Ensure data is two-dimensional (stereo format), even if the file is mono
        if self.original_data.ndim == 1:
            self.original_data = np.column_stack((self.original_data, self.original_data))
    
    def apply_delay(self, delay_seconds, feedback=0.5, mix=0.5):
        # Calculate the number of samples to delay
        delay_samples = int(delay_seconds * self.samplerate)
                
        # Create an output array, initialized to zero with the same shape as the input
        output = np.zeros_like(self.original_data)
        
        # Copy original audio data to output
        np.copyto(output, self.original_data)
        
        # Apply the delay effect
        for i in range(delay_samples, len(self.original_data)):
            
            # Mix the delayed output with feedback into the current position
            output[i] += (output[i - delay_samples] * feedback) * mix
        
        return output * mix + self.original_data * (1 - mix)

    def apply_distortion(self, threshold=0.5, gain=20):
        data = self.original_data.copy()

        # Normalise the original data to the range [-1, 1]
        max_val = np.max(np.abs(data))
        data /= max_val

        # Apply gain to the input signal
        data *= gain

        # Clipping des Signals
        data = np.clip(data, -threshold, threshold)

        # Bring the signal back to the original maximum amplitude
        data *= (max_val / np.max(np.abs(data)))

        return data

    
     
        
    
    def apply_soft_clipping(self, gain=20, threshold=0.8):
        data = self.original_data.copy()
        
        # Get the maximum absolute value from the audio data for normalization
        max_val = np.max(np.abs(data))
        
        # Normalize the data to ensure the peak is at 1
        data /= max_val
        
        # Increase the amplitude of the data by the gain factor
        data *= gain
        
        # Apply soft clipping to values above the positive threshold
        data = np.where(data > threshold, threshold + (data - threshold) / (1 + ((data - threshold) / (1 - threshold))**2), data)
        
        # Apply soft clipping to values below the negative threshold
        data = np.where(data < -threshold, -threshold + (data + threshold) / (1 + ((-data - threshold) / (1 - threshold))**2), data)
        
        # Rescale the data back to the original amplitude range
        data *= max_val
        return data

    def apply_sigmoid_distortion(self, gain=20):
        data = self.original_data.copy()
        
        # Normalize the audio data to the range [-1, 1]
        max_val = np.max(np.abs(data))
        data /= max_val
        
        # Apply the sigmoid function to distort the data
        # The sigmoid function transforms the data range to [0,1], then shifts it to [-1,1]
        data = 2 / (1 + np.exp(-gain * data)) - 1
        
        # Rescale the distorted data back to the original amplitude range
        data *= max_val
        return data

    def apply_chorus(self, depth_ms=20, decay=0.4):
        data = self.original_data.copy()
        
        # Calculate the number of samples to delay
        delay_samples = int(self.samplerate * depth_ms / 1000.0)
        
        # Create a low-pass filter using FIR window method
        filter_coeffs = scipy.signal.firwin(numtaps=30, cutoff=0.5, window="hamming")
        
        # Apply the filter to the data to get the delayed signal
        delayed_data = scipy.signal.lfilter(filter_coeffs, 1.0, data, axis=0)
        
        # Delay the filtered data by rolling it forward by the calculated number of samples
        delayed_data = np.roll(delayed_data, delay_samples, axis=0)
        
        # Mix the original data with the delayed data using the decay factor
        data = data * (1 - decay) + delayed_data * decay
        return data
    
    def apply_reverb(self, room_size=1, decay_factor=0.5, wet_level=0.33, dry_level=0.4, width=1.0, freeze_mode=0.0):
        data = self.original_data.copy()

        impulse_response_length = int(self.samplerate * room_size)
        impulse_response = np.random.randn(impulse_response_length)

        # Freeze mode control
        if freeze_mode != 0.0:
            impulse_response[:] = 1  # Setting all impulse response values to constant

        # Damping (exponential decay)
        impulse_response *= np.exp(-decay_factor * np.arange(impulse_response_length) / self.samplerate)


        # Abklingzeit einführen, um realistischeren Reverb zu erzeugen
        impulse_response *= np.exp(-decay_factor * np.arange(impulse_response_length) / self.samplerate)
        
        # Normalisieren der Impulsantwort, um Überamplifizierung zu verhindern
        impulse_response /= np.max(np.abs(impulse_response))
        
        # Apply convolution per channel to handle stereo signals correctly
        if data.ndim == 2:
            reverbed_data = np.zeros_like(data)
            for i in range(data.shape[1]):  # Assuming second dimension is channels
                reverbed_data[:, i] = scipy.signal.convolve(data[:, i], impulse_response, mode='full')[:len(data)]
        # Width handling
                if width != 1.0:
                    pan = (width - 0.5) * 2
                    reverbed_data[:, i] *= pan
        else:
            reverbed_data = scipy.signal.convolve(data, impulse_response, mode='full')[:len(data)]
        
        # Mix dry and wet signals
        mixed_data = dry_level * self.original_data + wet_level * reverbed_data


        # Normalize the output to prevent clipping
        max_val = np.max(np.abs(mixed_data))
        if max_val > 0:
            mixed_data /= max_val

        return mixed_data
    
    def save_audio(self,data, output_filename):
        # Mono or stereo is determined by channel count in the data
        sf.write(output_filename, data, self.samplerate)

    def play_audio(self, audio_data=None):
        if audio_data is None:
            audio_data = self.original_data
        sd.play(audio_data, self.samplerate)
        sd.wait()

 
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
        
        return output

    def apply_distortion(self, threshold=0.5):
        data = self.original_data
        max_val = np.max(np.abs(data)) 
        data = np.clip(data, -max_val * threshold, max_val * threshold)
        return data

    def apply_soft_clipping(self, gain=10, threshold=0.8):
        data = self.original_data.copy()
        max_val = np.max(np.abs(data))
        data /= max_val
        data *= gain
        data = np.where(data > threshold, threshold + (data - threshold) / (1 + ((data - threshold) / (1 - threshold))**2), data)
        data = np.where(data < -threshold, -threshold + (data + threshold) / (1 + ((-data - threshold) / (1 - threshold))**2), data)
        data *= max_val
        return data

    def apply_sigmoid_distortion(self, gain=20):
        data = self.original_data.copy()
        max_val = np.max(np.abs(data))
        data /= max_val
        data = 2 / (1 + np.exp(-gain * data)) - 1
        data *= max_val
        return data

    def apply_chorus(self, depth_ms=20, decay=0.4):
        data = self.original_data.copy()
        delay_samples = int(self.samplerate * depth_ms / 1000.0)
        filter_coeffs = scipy.signal.firwin(numtaps=30, cutoff=0.5, window="hamming")
        delayed_data = scipy.signal.lfilter(filter_coeffs, 1.0, data, axis=0)
        delayed_data = np.roll(delayed_data, delay_samples, axis=0)
        data = data * (1 - decay) + delayed_data * decay
        return data
    
    def apply_reverb(self, room_size=1):
        data = self.original_data.copy()
        impulse_response = np.random.randn(int(self.samplerate * room_size))
        
        # Apply convolution per channel to handle stereo signals correctly
        if data.ndim == 2:
            reverbed_data = np.zeros_like(data)
            for i in range(data.shape[1]):  # Assuming second dimension is channels
                reverbed_data[:, i] = scipy.signal.convolve(data[:, i], impulse_response, mode='full')[:data.shape[0]]
        else:
            reverbed_data = scipy.signal.convolve(data, impulse_response, mode='full')[:len(data)]
        
        return reverbed_data
    
    def save_audio(self,data, output_filename):
        # Mono or stereo is determined by channel count in the data
        sf.write(output_filename, data, self.samplerate)

    def play_audio(self, audio_data=None):
        if audio_data is None:
            audio_data = self.original_data
        sd.play(audio_data, self.samplerate)
        sd.wait()

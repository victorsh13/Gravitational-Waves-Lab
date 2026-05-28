import numpy as np
from scipy.signal import spectrogram

"""
FeatureExtractor class for extracting features from a dataset of signals. 
Given time-domain signals, this class transforms them into useful feature representations for analysis or ML tasks.
"""
class FeatureExtractor:
    def __init__(
        self,
        fs: int = 1024,
        n_window: int = 128,
        n_overlap: int = 64,
    ):
        """
        Initialize a FeatureExtractor instance.

        Parameters:
        fs (int): Sampling frequency of the input signal in Hz (default is 1024).
        n_window (int): Length of each window segment for Short-Time Fourier Transform (STFT) (default is 128).
        n_overlap (int): Number of points to overlap between segments (default is 64).

        Raises:
            ValueError: If length, fs, n_window, or n_overlap are invalid.
        """
        # Validate input parameters
        if fs <= 0:
            raise ValueError("Sampling frequency fs must be positive.")
        if not (0 < n_window): 
            raise ValueError("Segment length n_window must be positive.")
        if not (0 <= n_overlap < n_window):
            raise ValueError("Overlap n_overlap must be non-negative and less than segment length n_seg")
        
        # Initialize instance variables
        self.fs = fs
        self.n_window = n_window
        self.n_overlap = n_overlap

    
    def build_spectrogram(self,signal):
        """ 
        Compute the spectrogram of a signal using the Short-Time Fourier Transform (STFT).
        
        The spectrogram provides a time-frequency representation of the signal, showing how the frequency content evolves over time.

        Parameters:
        - signal: Input signal as a 1D numpy array.
        
        Returns:
        - f: Array of sample frequencies of size n_frequencies.
        - t_spec: Array of segment times centers of size n_time_segments.
        - Sxx_log: log-magnitude spectrogram matrix of the signal with dimensions (n_frequencies, n_time_segments).
        """
        signal = np.asarray(signal) # Ensure input is a numpy array for consistent processing

        if signal.ndim != 1:
            raise ValueError("Input signal must be a 1D array.")
        if signal.shape[0] == 0:
            raise ValueError("Input signal cannot be empty.")
        if self.n_window > len(signal):
            raise ValueError("Segment length n_window must be smaller than signal length.")
        


        f, t_spec, Sxx = spectrogram(
        signal,
        fs=self.fs,
        nperseg=self.n_window,
        noverlap=self.n_overlap,
        scaling='spectrum', #Use a scaling that normalizes the power of the spectrogram
        mode='magnitude' #Maginitude of the STFT. If 'psd' is used, it returns the power spectral density, which is approximately the square of the magnitude.
        )
        
        # We take the logarithm of the spectrogram to enhance visibility of features, especially for lower amplitude components.
        eps = 1e-12  # Small constant to prevent taking the logarithm of zero
        Sxx_db = 20 * np.log10(Sxx + eps) # Convert to relative decibels (dB), conventionally used in signal processing to represent the magnitude of the spectrogram on a logarithmic scale. 
                                        # If 'psd' mode is used, we would use 10 * np.log10(Sxx + eps) instead, since power is proportional to the square of the magnitude.
        
        
        return f, t_spec, Sxx_db




    def build_spectrogram_dataset(self, signals): 
        """ 
        Compute the spectrograms for a dataset of signals.

        This function applies 'build_spectrogram' to each signal in the dataset, returning a list of spectrograms.
        -----------------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - signals: 2D array of shape (n_samples, signal_length) containing the input signals.

        Returns:
        - X_spec: 3D array of shape (n_samples, n_frequencies, n_time_segments) containing the spectrograms for each signal.
        - f: Array of sample frequencies (same for all signals).
        - t_spec: Array of segment times centers (same for all signals).
        """

        signals = np.asarray(signals) # Ensure input is a numpy array for consistent processing

        if signals.ndim != 2:
            raise ValueError("Input 'signals' must be a 2D array of shape (n_samples, length).")
        if signals.shape[0] == 0 or signals.shape[1] == 0:
            raise ValueError("Input 'signals' cannot be empty.")

        spectrograms = []

        for signal in signals:
            f, t_spec, Sxx_db = self.build_spectrogram(signal)
            spectrograms.append(Sxx_db)

        X_spec = np.array(spectrograms) # Convert list of spectrograms to a 3D numpy array of shape (n_samples, n_frequencies, n_time_segments) 

        # If all the spectrograms doesn't have the same shape, this could be used:
        # shapes = [spec.shape for spec in spectrograms]
        # if len(set(shapes)) != 1:
        #   raise ValueError("All spectrograms must have the same shape.")
        # X_spec = np.stack(spectrograms), this way we would have to check if all the spectrograms have the same shape


        return X_spec, f, t_spec

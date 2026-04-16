import numpy as np
from .config import SimulationConfig
from pycbc.types.timeseries import TimeSeries



class SignalProcessor:
    def __init__(
            self, 
            config: SimulationConfig, 
            apply_whitening: bool = False, 
            apply_lowpass: bool = False, 
            apply_highpass: bool = False, 
            apply_standardization: bool = False, 
            preserve_length: bool = True, # Not implemented yet the False case
            lowpass_frequency: float = 350.0, 
            highpass_frequency: float = 30.0, 
            rng = None,
            ):
        
        if rng is None:
            rng = np.random.default_rng()

        self.config = config
        self.apply_whitening = apply_whitening
        self.apply_lowpass = apply_lowpass
        self.apply_highpass = apply_highpass
        self.apply_standardization = apply_standardization
        self.preserve_length = preserve_length
        self.lowpass_frequency = lowpass_frequency 
        self.highpass_frequency = highpass_frequency 


    def process(self, strain: TimeSeries) -> TimeSeries:
        """
        Process the given strain using the configuration set in the constructor.

        The processing includes:
            1. Whitening the strain to remove noise and improve SNR.
            2. Apply a low-pass filter to remove frequencies higher than lowpass_frequency.
            3. Apply a high-pass filter to remove frequencies lower than highpass_frequency.
            4. Standardize the strain to zero mean and unit variance.
            5. Restore the original length of the strain.
        
        Parameters
        ----------
        strain : TimeSeries
            The strain to be processed.
        
        Returns
        -------
        TimeSeries
            The processed strain.
        """
        processing_strain = strain.copy()
        original_length = len(processing_strain)
        original_start_time = processing_strain.start_time

        if self.apply_whitening:
            # Whiten the strain to remove noise and improve SNR
            # The whitening process is done using the PSD of the strain and a segment of the same duration, with a maximum filter duration of 1/3 of the segment duration.
            # The PSD is estimated using Welch's method, 
            processing_strain = processing_strain.whiten(
                segment_duration=processing_strain.get_duration(), # Duracion para estimar el PSD
                max_filter_duration=processing_strain.get_duration() / 3, # Larger filter better resolution in frequency but more corrupted edges, if shorter less damage in edges but worse resolution/stability in the filter.
                trunc_method='hann', # No se muy bien que es esto y que poner
                remove_corrupted=True,
                low_frequency_cutoff=self.highpass_frequency,
                return_psd=False
            )
              
        if self.apply_lowpass:
            # Apply a low-pass filter to remove frequencies higher than lowpass_frequency
            processing_strain = processing_strain.lowpass_fir(self.lowpass_frequency, order=512, beta=0.5, remove_corrupted=True) # Remove frequencies higher than lowpass_frequency
        if self.apply_highpass:
            # Apply a high-pass filter to remove frequencies lower than highpass_frequency
            processing_strain = processing_strain.highpass_fir(self.highpass_frequency, order=512, beta=0.5, remove_corrupted=True) # Remove frequencies lower than highpass_frequency
    
        if self.apply_standardization: # Standardization changes the absolute scale of the signal
            # Standardization is done by subtracting the mean and dividing by the standard deviation
            # The mean and standard deviation are calculated using the entire signal

            delta_t = processing_strain.delta_t
            epoch = processing_strain.start_time

            processing_array = np.asarray(processing_strain)
            mean = np.mean(processing_array)
            std_dev = np.std(processing_array)
            if std_dev == 0:
                processing_array = np.zeros_like(processing_array) # or zeros, not sure yet
            else:
                processing_array = (processing_array - mean) / std_dev
            processing_strain = TimeSeries(processing_array,
                                           delta_t=delta_t,
                                           epoch=epoch)

        if self.preserve_length:
            # Preserve the length of the original signal by zero-padding
            new_length = len(processing_strain) 
            diff_length = original_length - new_length
            if diff_length < 0:
                raise ValueError("Processed strain is longer than the original strain, cannot preserve length by zero-padding.")
            left_pad = diff_length // 2
            right_pad = diff_length - left_pad 

            processing_strain.prepend_zeros(left_pad)
            processing_strain.append_zeros(right_pad)
            processing_strain.start_time = original_start_time #We set the start time to the original start time, to align it with the duration of the original signal
            

        return processing_strain

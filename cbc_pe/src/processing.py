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
            preserve_length: bool = False, 
            lowpass_frequency: float = 350.0, 
            highpass_frequency: float = 30.0, 
            ):
        
        self.config = config
        self.apply_whitening = apply_whitening
        self.apply_lowpass = apply_lowpass
        self.apply_highpass = apply_highpass
        self.apply_standardization = apply_standardization
        self.preserve_length = preserve_length
        self.lowpass_frequency = lowpass_frequency 
        self.highpass_frequency = highpass_frequency 


    def process(self, strain: TimeSeries) -> TimeSeries:
        processing_strain = strain.copy()
        original_length = len(processing_strain)

        if self.apply_whitening:
            processing_strain = processing_strain.whiten(
                segment_duration=processing_strain.get_duration(), # Duracion para estimar el PSD
                max_filter_duration=processing_strain.get_duration() / 3, # Larger filter better resolution in frequency but more corrupted edges, if shorter less damage in edges but worse resolution/stability in the filter.
                trunc_method='hann', # No se muy bien que es esto y que poner
                remove_corrupted=True,
                low_frequency_cutoff=self.highpass_frequency,
                return_psd=False
            )
            
        if self.apply_lowpass:
            processing_strain = processing_strain.lowpass_fir(self.lowpass_frequency, order=512, beta=0.5, remove_corrupted=True) # Remove frequencies higher than lowpass_frequency
        if self.apply_highpass:
            processing_strain = processing_strain.highpass_fir(self.highpass_frequency, order=512, beta=0.5, remove_corrupted=True) # Remove frequencies lower than highpass_frequency    

       # if self.apply_standardization:
       #    processing_strain = processing_strain.standardize()

        return processing_strain

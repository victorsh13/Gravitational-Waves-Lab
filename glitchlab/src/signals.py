import numpy as np
from typing import Tuple


class SignalGenerator():

    ### CLASS INITIALIZATION ###
    def __init__(
        self,
        length=1024, #Default values 
        fs=1024,
        noise_std=0.1,
        seed=None,
        glitch_param_ranges=None,
    ):
        
        # -------------------------------- ----------------------------------- #
        # General information for generating glitches, including labels and generators  #
        # -------------------------------------------------------------------- #
        ### TO DO (Muy en el futuro): Crear un archivo json o similar 
        # donde se pueda configurar la informacion para generar los glitches
        """
        Initialize the SignalGenerator object.

        Parameters:
        length (int): The length of the signal to be generated. Defaults to 1024.
        fs (int): The sampling frequency of the signal. Defaults to 1024.
        noise_std (float): The standard deviation of the noise signal. Defaults to 0.1.
        seed (int or None): The seed for the random number generator. Defaults to None.
        glitch_param_ranges (dict or None): A dictionary with the glitch type as keys and the parameter ranges as values. If None, the default parameter ranges are used.

        Notes:
        The default parameter ranges for the glitches are as follows:
        - Sine Gaussian glitch: f0=[20, 80], sigma=[0.01, 0.15], amplitude=[0.5, 2.0]
        - Ringdown glitch: f0=[20, 80], tau=[0.02, 0.15], amplitude=[0.5, 2.0]

        The probabilities for the glitches are as follows:
        - Noise: 50%
        - Sine Gaussian glitch: 25%
        - Ringdown glitch: 25%

        The glitch type, generator, and probability can be accessed using the self.glitch_info dictionary.
        """
        self.glitch_info = {
            "noise": {
                'label': 0,
                'generator': self.gaussian_noise,
                'probability': 0.50,
                'param_ranges': None,
            },
            "sine_gaussian": {
                'label': 1,
                'generator': self.sine_gaussian_glitch,
                'probability': 0.25,
                'param_ranges': {
                    "f0": {"min": 20, "max": 80}, # Main frecuency in Hz
                    "sigma": {"min": 0.01, "max": 0.15}, #Width of sigma in seconds
                    "amplitude": {"min": 0.5, "max": 2.0}, #Amplitude  
                },
            },
            "ringdown": {
                'label': 2,
                'generator': self.ringdown_glitch,
                'probability': 0.25,
                'param_ranges': {
                    'f0': {"min": 20, "max": 80}, # Main frecuency in Hz
                    'tau': {"min": 0.02, "max": 0.15}, # Decay time in seconds
                    'amplitude': {"min": 0.5, "max": 2.0}, #Amplitude
                },
            }, 
            
        }

        
        # ------------------------- #
        # Validate input parameters #
        # ------------------------- #
        if not isinstance(length, int) or length <= 0:
            raise ValueError("'length' must be a positive integer.")
        if fs <= 0:
            raise ValueError("Sampling frequency (fs) must be positive.")
        if noise_std < 0:
            raise ValueError("Noise standard deviation (noise_std) must be non-negative.")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer or None.")
        if glitch_param_ranges is not None and not isinstance(glitch_param_ranges, dict):
            raise ValueError("glitch_param_ranges must be a dictionary or None.")
        
        # --------------------------------------------- #
        # Override glitch range parameters, if provided #
        # --------------------------------------------- #
        if glitch_param_ranges is not None and isinstance(glitch_param_ranges, dict):
            for glitch_type, param_ranges in glitch_param_ranges.items():
                if glitch_type not in self.glitch_info:
                    raise ValueError(f"Unknown glitch type: {glitch_type}"
                                    f"Supported types: {list(self.glitch_info.keys())}")
                if glitch_type == "noise":
                    raise ValueError(f"Glitch type 'noise' does not have glitch parameters to override.")
                self.glitch_info[glitch_type]["param_ranges"] = param_ranges

        # ------------------- #
        # Store configuration #
        # ------------------- #
        self.length = length
        self.fs = fs
        self.t = np.arange(length) / fs
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
 


    def sample_glitch_type(self) -> str:
        """
        Return a random glitch type with probability given in the dictionary.

        Returns:
            A glitch type
        """
        glitch_types = list(self.glitch_info.keys())
        probabilities = [self.glitch_info[glitch_type]["probability"] for glitch_type in glitch_types]

        # Check that probabilities sum to one
        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("Sum of probabilities must be 1.")

        # Sample a random glitch type
        return self.rng.choice(glitch_types, p=probabilities)

    def sample_glitch_params(self, glitch_type):
        """
        Return:
            A dictionary containing the initial parameters for generating a specified glitch type.
        """
        if glitch_type not in self.glitch_info:
            raise ValueError(f"Unknown glitch type: {glitch_type}"
                             f"Supported types: {list(self.glitch_info.keys())}")
        
        param_dict = {}
        if glitch_type == 'noise':
            param_dict=None
        else:
            for param in self.glitch_info[glitch_type]['param_ranges']:
                param_dict[param] = self.rng.uniform(self.glitch_info[glitch_type]['param_ranges'][param]["min"], self.glitch_info[glitch_type]['param_ranges'][param]["max"])
        
        return param_dict

    def gaussian_noise(self, mean: float = 0.0, std_dev: float = 1.0) -> np.ndarray:
        """
        Generate Gaussian noise.

        With mean = 0.0 and standard deviation = 1.0 by default, this function generates standard normal noise.
        """
        if not isinstance(self.length, int) or self.length <= 0:
            raise ValueError("Length of the noise signal must be a positive integer.")
        if std_dev < 0:
            raise ValueError("Standard deviation must be non-negative.")

        return self.rng.normal(loc=mean, scale=std_dev, size=self.length)

    def normalize_signal(self, signal):
        """ 
        Normalize a signal to have zero mean and unit variance.
        
        This is a common preprocessing step in signal processing and machine learning,
        ensuring that the signal has a consistent scale and distribution.
        """
        
        mean = np.mean(signal)
        std = np.std(signal)
        eps = 1e-8  # Small constant to prevent division by zero
        
        return (signal - mean) / (std + eps)  # Normalize to zero mean and unit variance


### -----------------------------------  GLITCHES DEFINITION ------------------------------------ 
#### TO DO: Include a glitch_SNR parameter to control the relative strength of the glitch compared to the noise
# which would allow to generate samples with varying levels of difficulty for classification tasks.
# SRN can be calculated as 20 * log10(amplitude / noise_std) and adjust the amplitude of the glitch accordingly based on the desired SNR level.

    def sine_gaussian_glitch(
        self,
        initial_glitch_params: dict
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate a sinusoidal Gaussian glitch.

        This function creates a glitch that is a product of a sinusoidal wave and a Gaussian envelope.
        The glitch is centered at time t0 and has a width (sigma) that controls how quickly it decays.

        Parameters:
        - self: Instance of the SignalGenerator class.
        - initial_glitch_params: A dictionary containing the initial parameters for generating the glitch:
            -> f0: Glitch frequency, sinusoidal component (Hz).
            -> sigma: Width of the Gaussian envelope (seconds).
            -> amplitude: Peak amplitude of the glitch.

        Returns:
        - glitch: The generated glitch signal as a numpy array.
        - glitch_params: A dictionary containing the parameters used to generate the glitch, useful for metadata.
        """

        # Unpack the initial glitch parameters
        f0 = initial_glitch_params['f0']
        sigma = initial_glitch_params['sigma']
        amplitude = initial_glitch_params['amplitude']
        

        if self.fs <= 0:
            raise ValueError("Sampling frequency must be positive.")
        if f0 <= 0 or f0 >= self.fs / 2:
            raise ValueError("Glitch frequency must be positive and less than Nyquist frequency (sampling_frequency/2).")
        if sigma <= 0:
            raise ValueError("Glitch width must be positive.")
        if amplitude < 0:
            raise ValueError("Glitch amplitude must be non-negative.")

    
        t0 = self.rng.uniform(self.t.min() + 3 * sigma, self.t.max() - 3 * sigma)  # Randomize the start time in a window to have at least 3 sigma of signal before starting or ending
        gaussian_envelope = np.exp(-0.5 * (self.t - t0) ** 2 / sigma ** 2)
        sinusoid = np.sin(2 * np.pi * f0 * (self.t - t0))
        glitch = amplitude * gaussian_envelope * sinusoid

        # Update the glitch parameters after generating the signal (include)
        glitch_params = {
            'f0': f0,
            'sigma': sigma,
            'amplitude': amplitude,
            't0': t0, 
        }

        return glitch, glitch_params

    def ringdown_glitch(
        self,
        initial_glitch_params: dict
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate a ringdown glitch signal.

        A ringdown glitch is a transient signal characterized by a sinusoidal component
        with an exponential decay. The signal is defined by four parameters: the frequency
        of the sinusoidal component, the decay time, the amplitude of the signal, and
        the time at which the glitch starts (t0).

        Parameters:
        - self: Instance of the SignalGenerator class.
        - initial_glitch_params: A dictionary containing the initial parameters for generating the glitch:
            -> f0: Frequency of the sinusoidal component (Hz).
            -> tau: Decay time (seconds).
            -> amplitude: Peak amplitude of the glitch.

        Returns:
        - glitch: The generated glitch signal as a numpy array.
        - glitch_params: A dictionary containing the parameters used to generate the glitch, useful for metadata.
        """

        # Unpack the initial glitch parameters
        f0 = initial_glitch_params['f0']
        tau = initial_glitch_params['tau']
        amplitude = initial_glitch_params['amplitude']

        if f0 <= 0 or f0 >= self.fs / 2:
            raise ValueError("Frequency must be positive and less than Nyquist frequency (sampling_frequency/2).")
        if tau <= 0:
            raise ValueError("Decay time must be positive.")
        if amplitude < 0:
            raise ValueError("Amplitude must be non-negative.")
        
        max_signal_window = self.t.max() - 3 * tau  # We take 3*tau to ensure that the glitch has enough time to decay before the end of the signal.
        t0  = self.rng.uniform(0.05 * self.t.max(), max_signal_window) # Start the glitch at least at 5% of the signal time

        time_mask = np.where(self.t > t0, 1, 0)  # Mask to ensure the glitch starts at start_time and is zero before that time
        exponential_decay = np.exp(-(self.t - t0) / tau)
        sinusoid = np.sin(2 * np.pi * f0 * (self.t - t0))
        glitch = (amplitude * sinusoid * exponential_decay) * time_mask

        glitch_params = {
            'f0': f0,
            'tau': tau,
            'amplitude': amplitude,
            't0': t0,
        }

        return glitch, glitch_params

    def generate_sample(self, glitch_type: str = None, glitch_params: dict = None) -> Tuple[np.ndarray, int, dict]:
        """
        Generate a sample signal with or without a glitch.

        This function creates a signal that may contain a glitch based on the specified parameters.
        It adds Gaussian noise to the signal and optionally includes a glitch of a specified type and parameters.

        Parameters:
        - glitch_type: The type of glitch to add to the signal. If None, a random glitch type is selected.
        - glitch_params: A dictionary containing the parameters for the selected glitch type.

        Returns:
        - signal: The generated signal as a numpy array.
        - label: An integer label indicating the presence of a glitch (1 for glitch, 0 for no glitch).
        - metadata: A dictionary containing information about the generated signal.
        """

        # Select a random glitch type if not provided
        glitch_type = self.sample_glitch_type() if glitch_type is None else glitch_type
        if glitch_type not in self.glitch_info:
            raise ValueError(f"Glitch type '{glitch_type}' is not supported. Supported types: {list(self.glitch_info.keys())}")

        # Select glitch parameters if not provided
        glitch_params = self.sample_glitch_params(glitch_type) if glitch_params is None else glitch_params

        # Generate noise
        noise = self.gaussian_noise(mean=0, std_dev=self.noise_std)

        # Generate glitch if required
        if glitch_type != 'noise':
            # Get the generator for the selected glitch type
            glitch_generator = self.glitch_info[glitch_type]['generator']

            # Generate the glitch
            glitch, glitch_metadata = glitch_generator(glitch_params)

            # Add the glitch to the noise
            signal = noise + glitch
        else:
            # If no glitch is required, just use the noise
            signal = noise
            glitch_metadata = None

        # Get the label for the glitch type
        label = self.glitch_info[glitch_type]['label']

        # Create metadata dictionary
        metadata = {
            'glitch_type': glitch_type,
            'glitch_metadata': glitch_metadata,
        }

        return signal, label, metadata

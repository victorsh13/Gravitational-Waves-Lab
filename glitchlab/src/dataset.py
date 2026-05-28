import numpy as np
from signals import SignalGenerator

class DatasetGenerator:
    def __init__(
        self,
        signal_generator=None,
        n_samples=100,
    ):
        
        # Create a SignalGenerator instance if not provided
        if signal_generator is None:
            signal_generator = SignalGenerator()
        
        # ------------------------- #
        # Validate input parameters #
        # ------------------------- #
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")
        if not isinstance(signal_generator, SignalGenerator):
            raise TypeError("'signal_generator' must be an instance of SignalGenerator.")

        # ------------------- #
        # Assign input values #
        # ------------------- #
        self.signal_generator = signal_generator
        self.n_samples = n_samples

    def generate_dataset(
        self,
        glitch_type: str = None,
        glitch_params: dict = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """    
        Generate a dataset of synthetic signals with and without glitches for classification or analysis tasks.

        Returns:
        - signals: Array of generated signals of shape (n_samples, length).
        - t: Time array of shape (length,).
        - labels: Array of labels (1 for glitch, 0 for noise) of shape (n_samples,).
        - metadata_list: List of dictionaries containing metadata for each sample.
        """
        signals: list = []
        labels: list = []
        metadata_list: list = []
        
        # has_glitch_array = rng.binomial(n=1, size=n_samples, p=glitch_probability)# Generate samples w/wo glitches based on the specified probability
        for _ in range(self.n_samples):
            signal, label, metadata = self.signal_generator.generate_sample(
                glitch_type=glitch_type,
                glitch_params=glitch_params,
            ) 
            signals.append(signal)
            labels.append(label)
            metadata_list.append(metadata)

        return np.array(signals), self.signal_generator.t, np.array(labels), metadata_list

            
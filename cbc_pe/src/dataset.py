import numpy as np
from. config import SimulationConfig
from .parameters import CBCParameters
from .sampling import ParameterSampler
from .waveform import WaveformGenerator
from .detectors import DetectorProjector
from .noise import NoiseModel
from .injection import SignalInjector
from .processing import SignalProcessor
from .labels import LabelTransformer
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSample:
    X: np.ndarray
    y: np.ndarray
    parameters: CBCParameters

@dataclass(frozen=True)
class DatasetBatch:
    X: np.ndarray
    y: np.ndarray   
    parameters: list[CBCParameters]


class DatasetBuilder:

    def __init__(
        self,
        parameter_sampler: ParameterSampler,
        waveform_generator: WaveformGenerator,
        detector_projector: DetectorProjector,
        noise_model: NoiseModel,
        signal_injector: SignalInjector,
        signal_processor: SignalProcessor,
        label_transformer: LabelTransformer,
        detector_names: list[str] | None = None,
    ) -> None:
        """
        Initialize a DatasetBuilder object.

        Parameters
        ----------
        parameter_sampler : ParameterSampler
            The sampler for the parameters.
        waveform_generator : WaveformGenerator
            The generator for the waveforms.
        detector_projector : DetectorProjector
            The projector for the detectors.
        noise_model : NoiseModel
            The model for the noise.
        signal_injector : signal_injector
            The engine for injecting the signals into the noise.
        signal_processor : SignalProcessor
            The processor for the signals.
        label_transformer : LabelTransformer
            The transformer for the labels.
        detector_names : list[str]
            The names of the detectors to use.
        """
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        self.parameter_sampler = parameter_sampler
        self.waveform_generator = waveform_generator
        self.detector_projector = detector_projector
        self.noise_model = noise_model
        self.signal_injector = signal_injector
        self.signal_processor = signal_processor
        self.label_transformer = label_transformer
        self.detector_names = detector_names

        if not detector_names:
            raise ValueError("detector_names must contain at least one detector.")

    def build_sample(
        self,
        params: CBCParameters | None = None,
        standardize_labels: bool = False,
        injection_time: float | None = None,
    ) -> DatasetSample:
        """
        Build a sample of a dataset

        Parameters
        ----------
        params : CBCParameters
            The parameters of the binary compact object

        Returns
        -------
        DatasetSample
            A DatasetSample object with the content: (X, y, parameters)
            - X: The processed signal. The shape is (n_detectors, signal_length)
            - y: The label
            - parameters: The parameters of the binary compact object
        """

        # Sample a new set of parameters if not provided
        if params is None:
            params = self.parameter_sampler.sample_one()

        # Generate the waveform
        hp, hc = self.waveform_generator.generate(params)

        # Project the waveform onto the detector
        projected_waveform = self.detector_projector.project(hp, hc, params)

        channels = []

        # Iterate over the detectors
        for detector_name in self.detector_names:
            # Sample the noise
            noise = self.noise_model.sample(detector_name)

            # Inject the waveform into the noise
            injected_signal = self.signal_injector.inject(noise, projected_waveform[detector_name], injection_time=injection_time)

            # Process the signal
            processed_signal = self.signal_processor.process(injected_signal.strain)

            # Append the processed signal to the list
            channels.append(np.asarray(processed_signal))

        # Stack the channels and produce the labels
        X = np.stack(channels, axis=0)
        y = self.label_transformer.transform(params, standardize=standardize_labels)

        return DatasetSample(X=X, y=y, parameters=params)
    

    def build_dataset(
        self,
        num_samples: int,
        standardize_labels: bool = False,
        injection_time: float | None = None,
        ) -> DatasetBatch:

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        X_list = []
        y_list = []
        parameters_list = []

        for _ in range(num_samples):
            sample = self.build_sample(standardize_labels=standardize_labels, injection_time=injection_time)
            X_list.append(sample.X)
            y_list.append(sample.y)
            parameters_list.append(sample.parameters)

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        return DatasetBatch(X=X, y=y, parameters=parameters_list)
    
    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        detector_names: list[str] | None = None,
        signal_processor_kwargs: dict | None = None, # Dictionary containing the kwargs for the signal processor.
        label_transformer_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Create a new instance of the DatasetBuilder from a SimulationConfig object.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration object.
        detector_names : list[str] | None
            The list of detector names to use. If None, use all three detectors: H1, L1, and V1.
        signal_processor_kwargs : dict | None
            The dictionary containing the kwargs for the signal processor. If None, use the default kwargs.
        label_transformer_kwargs : dict | None
            The dictionary containing the kwargs for the label transformer. If None, use the default kwargs.
        rng : np.random.Generator | None
            The random number generator to use. If None, use the default random number generator.

        Returns
        -------
        DatasetBuilder
            The new instance of the DatasetBuilder.
        """

        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        if signal_processor_kwargs is None:
            signal_processor_kwargs = {}

        if label_transformer_kwargs is None:
            label_transformer_kwargs = {}

        rng = rng if rng is not None else np.random.default_rng()

        parameter_sampler = ParameterSampler(rng=rng)
        waveform_generator = WaveformGenerator(config=config)
        detector_projector = DetectorProjector(detector_names=detector_names)
        noise_model = NoiseModel(config=config)
        signal_injector = SignalInjector(config=config, rng=rng)
        signal_processor = SignalProcessor(config=config, **signal_processor_kwargs)
        label_transformer = LabelTransformer(**label_transformer_kwargs)

        return cls(
            parameter_sampler=parameter_sampler,
            waveform_generator=waveform_generator,
            detector_projector=detector_projector,
            noise_model=noise_model,
            signal_injector=signal_injector,
            signal_processor=signal_processor,
            label_transformer=label_transformer,
            detector_names=detector_names,
        )
    

       
import numpy as np

from .config import SimulationConfig
from .parameters import CBCParameters
from .sampling import ParameterSampler
from .waveform import WaveformGenerator
from .detectors import DetectorProjector
from .noise import NoiseModel
from .injection import SignalInjector
from .processing import SignalProcessor
from .labels import LabelTransformer
from .snr import compute_detector_optimal_snr, compute_network_snr, rescale_distance_for_target_network_snr

from dataclasses import dataclass, replace
from pycbc.types.timeseries import TimeSeries
from warnings import warn


@dataclass(frozen=True)
class DatasetSample:
    X: np.ndarray
    y: np.ndarray
    parameters: CBCParameters
    injection_time: float | None = None
    network_snr: float | None = None

@dataclass(frozen=True)
class DatasetBatch:
    X: np.ndarray
    y: np.ndarray   
    parameters: list[CBCParameters]
    injection_times: list[float | None]
    network_snrs: list[float | None]


class DatasetBuilder:

    def __init__(
        self,
        config: SimulationConfig,
        parameter_sampler: ParameterSampler,
        waveform_generator: WaveformGenerator,
        detector_projector: DetectorProjector,
        noise_model: NoiseModel,
        signal_injector: SignalInjector,
        signal_processor: SignalProcessor,
        label_transformer: LabelTransformer,
        detector_names: list[str] | None = None,
        rng: np.random.Generator | None = None,
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

        self.rng = rng if rng is not None else np.random.default_rng()
        self.config = config
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
        injection_time : float | None
            The injection time of the signal. If None, use the default injection time.
        standardize_labels : bool
            Whether to standardize the labels. If True, the labels will be standardized to have a mean of 0 and a standard deviation of 1.
        rng : np.random.Generator | None
            The random number generator to use. If None, the default random number generator is used.
            
        Returns
        -------
        DatasetSample
            A DatasetSample object with the content: (X, y, parameters)
            - X: The processed signal. The shape is (n_detectors, signal_length)
            - y: The label
            - parameters: The parameters of the binary compact object
            - injection_time: The injection time of the signal in the geocentric reference frame
            - network_snr: The network SNR of the signal
        """

        # Sample a new set of parameters if not provided
        if params is None:
            params = self.parameter_sampler.sample_one()


        projected_waveform, current_network_snr = self._generate_projected_waveforms_and_snr(params)

        # Check if the network SNR is within the target network SNR range
        if self.config.target_network_snr_range is not None:
            target_network_snr_min, target_network_snr_max = self.config.target_network_snr_range
            if current_network_snr < target_network_snr_min or current_network_snr > target_network_snr_max:
                # Rescale the distance to achieve the target network SNR
                
                target_network_snr = self.rng.uniform(target_network_snr_min, target_network_snr_max)
                rescaled_distance = rescale_distance_for_target_network_snr(params.distance, current_network_snr, target_network_snr)

                params = replace(params, distance=rescaled_distance)
                projected_waveform, current_network_snr = self._generate_projected_waveforms_and_snr(params)

                # Check if the network SNR is within the target network SNR range assuming certain tolerance
                relative_snr_error = np.abs(current_network_snr - target_network_snr) / target_network_snr
                if relative_snr_error > self.config.snr_relative_tolerance:
                    warning = f"The network SNR is far from the target network SNR range after rescaling. Current network SNR: {current_network_snr:.2f}, Target network SNR: {target_network_snr:.2f}, Relative SNR error: {relative_snr_error:.2f}."
                    warn(warning)

       
        injection_time = self.set_injection_time(
            projected_waveforms=projected_waveform,
            injection_time=injection_time,
            rng=self.rng,
        )

        channels = []

        # Iterate over the detectors
        for detector_name in self.detector_names:
            # Sample the noise
            noise = self.noise_model.sample(detector_name)

            # Inject the waveform into the noise
            #detector_injection_time = injection_time + time_delays[detector_name]
            injected_signal = self.signal_injector.inject(noise, projected_waveform[detector_name], injection_time=injection_time)

            # Process the signal
            processed_signal = self.signal_processor.process(injected_signal.strain)

            # Append the processed signal to the list
            channels.append(np.asarray(processed_signal))

        # Stack the channels and produce the labels
        X = np.stack(channels, axis=0)
        y = self.label_transformer.transform(params, standardize=standardize_labels)

        return DatasetSample(
            X=X,
            y=y,
            parameters=params,
            injection_time=injection_time,
            network_snr=current_network_snr,
        )
    

    def build_dataset(
        self,
        num_samples: int,
        #params: CBCParameters | None = None,
        standardize_labels: bool = False,
        injection_time: float | None = None,
        max_attempts: int | None = None,
    ) -> DatasetBatch:

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        if max_attempts is None:
            max_attempts = 10 * num_samples

        X_list = []
        y_list = []
        parameters_list = []
        injection_times = []
        network_snrs = []

        n_attempts = 0
        n_failed = 0

        while len(X_list) < num_samples and n_attempts < max_attempts:
            n_attempts += 1

            try:
                sample = self.build_sample(
                    standardize_labels=standardize_labels,
                    injection_time=injection_time,
                )

            except ValueError as e:
                n_failed += 1

                # Para no llenar la salida con 1000 errores
                if n_failed <= 10:
                    print(f"Skipping failed sample attempt {n_attempts}: {e}")
                elif n_failed == 11:
                    print("Further failed sample messages will be suppressed...")

                continue

            X_list.append(sample.X)
            y_list.append(sample.y)
            parameters_list.append(sample.parameters)
            injection_times.append(sample.injection_time)
            network_snrs.append(sample.network_snr)

            n_done = len(X_list)

            if n_done % 10 == 0:
                print(
                    f"Built sample {n_done} of {num_samples} "
                    f"--> {n_done / num_samples:.1%} completed "
                    f"(attempts={n_attempts}, failed={n_failed})"
                )

        if len(X_list) < num_samples:
            raise RuntimeError(
                f"Could only build {len(X_list)} valid samples out of {num_samples} "
                f"after {n_attempts} attempts. Failed samples: {n_failed}. "
                f"Your simulation config is probably rejecting too many samples."
            )

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        print(
            f"--> DATASET GENERATED with {len(X_list)} valid samples "
            f"after {n_attempts} attempts. Failed samples: {n_failed}."
        )

        return DatasetBatch(
            X=X,
            y=y,
            parameters=parameters_list,
            injection_times=injection_times,
            network_snrs=network_snrs,
        )
    

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
        signal_processor = SignalProcessor(config=config, **signal_processor_kwargs, rng=rng)
        label_transformer = LabelTransformer(**label_transformer_kwargs)

        return cls(
            config=config,
            parameter_sampler=parameter_sampler,
            waveform_generator=waveform_generator,
            detector_projector=detector_projector,
            noise_model=noise_model,
            signal_injector=signal_injector,
            signal_processor=signal_processor,
            label_transformer=label_transformer,
            detector_names=detector_names,
        )
    

    def set_injection_time(
        self,
        projected_waveforms: dict[str, TimeSeries],
        injection_time: float | None = None,
        rng: np.random.Generator | None = None,
        ) -> float:
        """
        Choose a valid geocentric injection time for a set of projected detector waveforms.

        The returned injection_time is the geocentric reference time of the event
        within the noise segment. It is chosen on the discrete sample grid so that,
        after accounting for each waveform's detector-dependent start/end times,
        every projected waveform fits fully inside the strain segment.
        """
        if rng is None:
            rng = np.random.default_rng()

        delta_t = self.config.delta_t
        strain_start = 0.0
        strain_end = self.config.duration

        lower_bounds = []
        upper_bounds = []

        for detector_name, waveform in projected_waveforms.items():
            signal_start = float(waveform.start_time)
            signal_end = float(waveform.end_time)

            # Continuous-time valid interval for the geocentric reference time
            lower_bounds.append(strain_start - signal_start)
            upper_bounds.append(strain_end - signal_end)

        global_min = max(lower_bounds)
        global_max = min(upper_bounds)

        if global_min > global_max:
            raise ValueError(
                f"No valid geocentric injection time exists. "
                f"global_min={global_min:.6f}, global_max={global_max:.6f}"
            )

        # Use the central 50% of the valid interval as a safety margin
        safe_min = global_min + 0.25 * (global_max - global_min)
        safe_max = global_min + 0.75 * (global_max - global_min)

        # Convert the safe continuous interval to valid integer sample indices
        min_index = int(np.ceil(safe_min / delta_t))
        max_index = int(np.floor(safe_max / delta_t))

        if max_index < min_index:
            raise ValueError(
                "Safe injection interval is empty after discretization. "
                f"safe_min={safe_min:.6f}, safe_max={safe_max:.6f}, "
                f"min_index={min_index}, max_index={max_index}"
            )

        if injection_time is None:
            injection_index = rng.integers(min_index, max_index + 1)
            injection_time = injection_index * delta_t
        else:
            # Quantize user-provided time to the sample grid
            injection_index = int(round(injection_time / delta_t))
            injection_time = injection_index * delta_t

            if not (min_index <= injection_index <= max_index):
                raise ValueError(
                    f"injection_time={injection_time:.6f} s (index={injection_index}) "
                    f"is outside the valid discrete interval "
                    f"[{min_index * delta_t:.6f}, {max_index * delta_t:.6f}] s "
                    f"(indices [{min_index}, {max_index}])."
                )

        return injection_time
 


    def _generate_projected_waveforms_and_snr(self, params: CBCParameters) -> tuple[dict[str, TimeSeries], float]:
        """
        Compute the projected waveforms and the network SNR for a given set of parameters.
        """
        
        # Generate the waveform
        hp, hc = self.waveform_generator.generate(params)

        # Project the waveform onto the detector
        projected_waveform = self.detector_projector.project(hp, hc, params)

        network_snr = []
        for name, waveform in projected_waveform.items():
            detector_noise = self.noise_model.psds[name]
            optimal_snr =compute_detector_optimal_snr(waveform, detector_noise, self.config)
            network_snr.append(optimal_snr)

        return projected_waveform, compute_network_snr(np.array(network_snr))

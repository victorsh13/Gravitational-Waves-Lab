from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from warnings import warn

from pycbc.types.timeseries import TimeSeries

from .config import SimulationConfig
from .parameters import CBCParameters
from .sampling import ParameterSampler, PriorConfig
from .waveform import WaveformGenerator
from .windowing import WaveformWindowSelector
from .detectors import DetectorProjector
from .noise import NoiseModel
from .injection import SignalInjector, InjectionResult
from .processing import SignalProcessor
from .labels import LabelTransformer
from .snr import (
    compute_network_optimal_snr,
    decide_distance_rescaling,
    validate_snr_rescaling,
)


@dataclass(frozen=True)
class BuiltSignalNetwork:
    """
    Intermediate object containing the noiseless projected signal network
    embedded into fixed-duration zero segments.
    """

    params: CBCParameters
    waveform: object
    windowed: object
    projection: object
    placement: object
    signal_segment_results: dict[str, InjectionResult]
    signal_segments: dict[str, TimeSeries]
    detector_snrs: dict[str, float]
    network_snr: float


@dataclass(frozen=True)
class DatasetSample:
    """
    Single generated dataset sample.

    Attributes
    ----------
    X : np.ndarray
        Processed detector channels with shape (n_detectors, n_samples).
    y : np.ndarray
        Label vector.
    parameters : CBCParameters
        Final physical parameters used to generate the sample. If SNR rescaling
        was applied, this contains the rescaled distance.
    metadata : dict
        Generation metadata.
    """

    X: np.ndarray
    y: np.ndarray
    parameters: CBCParameters
    metadata: dict


@dataclass(frozen=True)
class DatasetBatch:
    """
    Batch of generated dataset samples.
    """

    X: np.ndarray
    y: np.ndarray
    parameters: list[CBCParameters]
    metadata: list[dict]


class DatasetBuilder:
    def __init__(
        self,
        config: SimulationConfig,
        parameter_sampler: ParameterSampler,
        waveform_generator: WaveformGenerator,
        waveform_window_selector: WaveformWindowSelector,
        detector_projector: DetectorProjector,
        noise_model: NoiseModel,
        signal_injector: SignalInjector,
        signal_processor: SignalProcessor,
        label_transformer: LabelTransformer,
        detector_names: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        if not detector_names:
            raise ValueError("detector_names must contain at least one detector.")

        self.config = config
        self.parameter_sampler = parameter_sampler
        self.waveform_generator = waveform_generator
        self.waveform_window_selector = waveform_window_selector
        self.detector_projector = detector_projector
        self.noise_model = noise_model
        self.signal_injector = signal_injector
        self.signal_processor = signal_processor
        self.label_transformer = label_transformer
        self.detector_names = detector_names
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        detector_names: list[str] | None = None,
        signal_processor_kwargs: dict | None = None,
        label_transformer_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> "DatasetBuilder":
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        if signal_processor_kwargs is None:
            signal_processor_kwargs = {}

        if label_transformer_kwargs is None:
            label_transformer_kwargs = {}

        rng = rng if rng is not None else np.random.default_rng()

        prior_config = cls._prior_config_from_simulation_config(config)

        parameter_sampler = ParameterSampler(rng=rng, prior_config=prior_config)
        waveform_generator = WaveformGenerator(config=config)
        waveform_window_selector = WaveformWindowSelector(config=config)
        detector_projector = DetectorProjector(detector_names=detector_names)
        noise_model = NoiseModel(config=config)
        signal_injector = SignalInjector(config=config, rng=rng)
        signal_processor = SignalProcessor(
            config=config,
            **signal_processor_kwargs,
            rng=rng,
        )
        label_transformer = LabelTransformer(**label_transformer_kwargs)

        return cls(
            config=config,
            parameter_sampler=parameter_sampler,
            waveform_generator=waveform_generator,
            waveform_window_selector=waveform_window_selector,
            detector_projector=detector_projector,
            noise_model=noise_model,
            signal_injector=signal_injector,
            signal_processor=signal_processor,
            label_transformer=label_transformer,
            detector_names=detector_names,
            rng=rng,
        )

    def build_sample(
        self,
        params: CBCParameters | None = None,
        standardize_labels: bool = False,
        geocentric_coalescence_time: float | None = None,
        placement_policy: str = "random_contained",
    ) -> DatasetSample:
        """
        Build one simulated dataset sample.

        Pipeline
        --------
        1. Sample or receive physical parameters.
        2. Generate full waveform.
        3. Select the usable waveform window.
        4. Project onto detectors using a geocentric reference time.
        5. Choose a common 4 s strain segment containing the projected network.
        6. Embed projected detector signals into zero-valued 4 s segments.
        7. Compute initial SNR.
        8. Optionally rescale distance to match target network SNR range.
        9. Generate noise and inject the final projected signals.
        10. Process detector channels.
        11. Build labels and metadata.
        """
        if params is None:
            params = self.parameter_sampler.sample_one()

        if geocentric_coalescence_time is None:
            geocentric_coalescence_time = self._sample_geocentric_coalescence_time()

        initial_network = self._build_projected_signal_network(
            params=params,
            geocentric_coalescence_time=geocentric_coalescence_time,
            placement_policy=placement_policy,
        )

        snr_decision = decide_distance_rescaling(
            current_distance=params.distance,
            current_network_snr=initial_network.network_snr,
            target_network_snr_range=self.config.target_network_snr_range,
            rng=self.rng,
        )

        if snr_decision.should_rescale:
            params = params.with_distance(snr_decision.new_distance)

            final_network = self._build_projected_signal_network(
                params=params,
                geocentric_coalescence_time=geocentric_coalescence_time,
                placement_policy=placement_policy,
            )

            try:
                validate_snr_rescaling(
                    final_network_snr=final_network.network_snr,
                    target_network_snr=snr_decision.target_network_snr,
                    relative_tolerance=self.config.snr_relative_tolerance,
                )
            except ValueError as exc:
                warn(str(exc))

        else:
            final_network = initial_network

        noises = self.noise_model.sample_network(
            detector_names=self.detector_names,
            seed=int(self.rng.integers(0, 2**32 - 1)),
        )

        noises = {
            detector_name: self.signal_injector.set_strain_start_time(
                strain=noise,
                start_time=final_network.placement.segment_start_time,
            )
            for detector_name, noise in noises.items()
        }

        injected_results = self.signal_injector.inject_network(
            noises=noises,
            signals=final_network.projection.strains,
        )

        channels: list[np.ndarray] = []

        for detector_name in self.detector_names:
            processed_signal = self.signal_processor.process(
                injected_results[detector_name].strain
            )
            channels.append(np.asarray(processed_signal))

        X = np.stack(channels, axis=0)

        y = self.label_transformer.transform(
            params,
            standardize=standardize_labels,
        )

        metadata = self._build_metadata(
            geocentric_coalescence_time=geocentric_coalescence_time,
            initial_network=initial_network,
            final_network=final_network,
            injected_results=injected_results,
            snr_decision=snr_decision,
            placement_policy=placement_policy,
        )

        return DatasetSample(
            X=X,
            y=y,
            parameters=params,
            metadata=metadata,
        )

    def build_dataset(
        self,
        num_samples: int,
        standardize_labels: bool = False,
        geocentric_coalescence_time: float | None = None,
        placement_policy: str = "random_contained",
        max_attempts: int | None = None,
    ) -> DatasetBatch:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        if max_attempts is None:
            max_attempts = 10 * num_samples

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        parameters_list: list[CBCParameters] = []
        metadata_list: list[dict] = []

        n_attempts = 0
        n_failed = 0

        while len(X_list) < num_samples and n_attempts < max_attempts:
            n_attempts += 1

            try:
                sample = self.build_sample(
                    standardize_labels=standardize_labels,
                    geocentric_coalescence_time=geocentric_coalescence_time,
                    placement_policy=placement_policy,
                )

            except (ValueError, RuntimeError) as exc:
                n_failed += 1

                if n_failed <= 10:
                    print(f"Skipping failed sample attempt {n_attempts}: {exc}")
                elif n_failed == 11:
                    print("Further failed sample messages will be suppressed...")

                continue

            X_list.append(sample.X)
            y_list.append(sample.y)
            parameters_list.append(sample.parameters)
            metadata_list.append(sample.metadata)

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
                "Your simulation config is probably rejecting too many samples."
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
            metadata=metadata_list,
        )

    def _build_projected_signal_network(
        self,
        params: CBCParameters,
        geocentric_coalescence_time: float,
        placement_policy: str,
    ) -> BuiltSignalNetwork:
        """
        Build the noiseless projected detector network and compute its SNR.

        The projected detector signals are embedded into fixed-duration
        zero-valued segments using the same placement policy later used for
        actual noise injection.
        """
        waveform = self.waveform_generator.generate(params)

        windowed = self.waveform_window_selector.select(
            waveform.h_plus,
            waveform.h_cross,
        )

        projection = self.detector_projector.project(
            h_plus=windowed.h_plus,
            h_cross=windowed.h_cross,
            parameters=params,
            geocentric_coalescence_time=geocentric_coalescence_time,
        )

        self._validate_detector_set(projection.strains)

        placement = self.signal_injector.choose_segment_placement_containing_network(
            signals=projection.strains,
            placement_policy=placement_policy,
        )

        zero_segments = {
            detector_name: self.signal_injector.build_zero_strain(
                start_time=placement.segment_start_time,
            )
            for detector_name in projection.strains
        }

        signal_segment_results = self.signal_injector.inject_network(
            noises=zero_segments,
            signals=projection.strains,
        )

        signal_segments = {
            detector_name: result.strain
            for detector_name, result in signal_segment_results.items()
        }

        psds = {
            detector_name: self.noise_model.get_psd(detector_name)
            for detector_name in signal_segments
        }

        detector_snrs, network_snr = compute_network_optimal_snr(
            signal_segments=signal_segments,
            psds=psds,
            config=self.config,
        )

        return BuiltSignalNetwork(
            params=params,
            waveform=waveform,
            windowed=windowed,
            projection=projection,
            placement=placement,
            signal_segment_results=signal_segment_results,
            signal_segments=signal_segments,
            detector_snrs=detector_snrs,
            network_snr=network_snr,
        )

    def _build_metadata(
        self,
        geocentric_coalescence_time: float,
        initial_network: BuiltSignalNetwork,
        final_network: BuiltSignalNetwork,
        injected_results: dict[str, InjectionResult],
        snr_decision,
        placement_policy: str,
    ) -> dict:
        return {
            "simulation": self._simulation_metadata(),
            "geocentric_coalescence_time": geocentric_coalescence_time,
            "detectors": list(self.detector_names),
            "placement_policy": placement_policy,
            "waveform": self._waveform_metadata(final_network),
            "windowing": self._safe_dataclass_to_dict(final_network.windowed.metadata),
            "projection": self._safe_dataclass_to_dict(final_network.projection.metadata),
            "placement": self._safe_dataclass_to_dict(final_network.placement),
            "snr": {
                "initial_detector_snrs": dict(initial_network.detector_snrs),
                "initial_network_snr": float(initial_network.network_snr),
                "final_detector_snrs": dict(final_network.detector_snrs),
                "final_network_snr": float(final_network.network_snr),
                "snr_rescaled": bool(snr_decision.should_rescale),
                "snr_rescaling_reason": snr_decision.reason,
                "target_network_snr": float(snr_decision.target_network_snr),
                "distance_before_rescale": float(snr_decision.old_distance),
                "distance_after_rescale": float(snr_decision.new_distance),
            },
            "injection": self._injection_metadata(injected_results),
            "noise": self.noise_model.metadata(),
        }

    def _waveform_metadata(self, network: BuiltSignalNetwork) -> dict:
        metadata = {}

        if hasattr(network.waveform, "metadata"):
            metadata["generated"] = self._safe_dataclass_to_dict(
                network.waveform.metadata
            )

        metadata["params_distance"] = float(network.params.distance)

        return metadata

    def _injection_metadata(
        self,
        injected_results: dict[str, InjectionResult],
    ) -> dict:
        output = {}

        for detector_name, result in injected_results.items():
            output[detector_name] = {
                "signal_start_time": float(result.signal_start_time),
                "signal_end_time": float(result.signal_end_time),
                "segment_start_time": float(result.segment_start_time),
                "segment_end_time": float(result.segment_end_time),
                "signal_start_index": int(result.signal_start_index),
                "signal_end_index": int(result.signal_end_index),
                "n_signal_samples": int(result.n_signal_samples),
                "n_injected_samples": int(result.n_injected_samples),
            }

        return output

    def _simulation_metadata(self) -> dict:
        output = {
            "sampling_frequency": float(self.config.sampling_frequency),
            "duration": float(self.config.duration),
            "delta_t": float(self.config.delta_t),
            "length": int(self.config.length),
            "delta_f": float(self.config.delta_f),
            "flength": int(self.config.flength),
            "low_frequency_cutoff": float(self.config.low_frequency_cutoff),
            "waveform_approximant": self.config.waveform_approximant,
            "target_network_snr_range": self.config.target_network_snr_range,
            "snr_relative_tolerance": float(self.config.snr_relative_tolerance),
            "truncation_policy": self.config.truncation_policy,
            "required_final_duration": float(self.config.required_final_duration),
        }

        for optional_attr in [
            "simulation_regime",
            "waveform_family",
            "event_time_reference",
            "snr_on_truncated_signal",
        ]:
            if hasattr(self.config, optional_attr):
                value = getattr(self.config, optional_attr)

                if isinstance(value, (float, int, str, bool, tuple, type(None))):
                    output[optional_attr] = value

        return output

    def _validate_detector_set(self, signals: dict[str, TimeSeries]) -> None:
        signal_detectors = set(signals.keys())
        expected_detectors = set(self.detector_names)

        if signal_detectors != expected_detectors:
            raise ValueError(
                "Projected signal detector set does not match builder detector_names. "
                f"signals={signal_detectors}, expected={expected_detectors}"
            )

    def _sample_geocentric_coalescence_time(self) -> float:
        """
        Return a geocentric coalescence time.

        For now this is fixed for reproducibility. Later this can be sampled
        over a GPS-time interval to vary antenna patterns with Earth's rotation.
        """
        return 1126259462.0

    @staticmethod
    def _safe_dataclass_to_dict(obj) -> dict:
        """
        Convert a metadata dataclass to dict.

        This should only be used for lightweight metadata dataclasses, not for
        objects containing TimeSeries arrays.
        """
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)

        if isinstance(obj, dict):
            return dict(obj)

        raise TypeError(f"Object of type {type(obj)} cannot be converted to dict.")
    
    @staticmethod
    def _prior_config_from_simulation_config(config: SimulationConfig) -> PriorConfig:
        regime = getattr(config, "simulation_regime", "BBH")

        if regime == "BBH":
            return PriorConfig.bbh()

        if regime == "BNS":
            return PriorConfig.bns()

        raise ValueError(f"Unsupported simulation_regime: {regime}")
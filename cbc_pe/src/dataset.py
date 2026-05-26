from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import time

from pycbc.types.timeseries import TimeSeries

from .config import SimulationConfig
from .parameters import CBCParameters
from .sampling import ParameterSampler, PriorConfig
from .waveform import WaveformGenerator
from .windowing import ProjectedNetworkWindowSelector
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
        network_window_selector: ProjectedNetworkWindowSelector,
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
        self.network_window_selector = network_window_selector
        self.detector_projector = detector_projector
        self.noise_model = noise_model
        self.signal_injector = signal_injector
        self.signal_processor = signal_processor
        self.label_transformer = label_transformer
        self.detector_names = detector_names
        self.rng = rng if rng is not None else np.random.default_rng()

        for detector_name in detector_names:
            self.noise_model.get_psd(detector_name)

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        detector_names: list[str] | None = None,
        signal_processor_kwargs: dict | None = None,
        label_transformer_kwargs: dict | None = None,
        parameter_sampler_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> "DatasetBuilder":
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        if signal_processor_kwargs is None:
            signal_processor_kwargs = {}

        if label_transformer_kwargs is None:
            label_transformer_kwargs = {}

        rng = rng if rng is not None else np.random.default_rng()

        prior_config = PriorConfig.from_dict(
            parameter_sampler_kwargs,
            default_regime=getattr(config, "simulation_regime", "BBH"),
        )

        parameter_sampler = ParameterSampler(
            rng=rng,
            prior_config=prior_config,
        )
        waveform_generator = WaveformGenerator(config=config)
        network_window_selector = ProjectedNetworkWindowSelector(config=config)
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
            network_window_selector=network_window_selector,
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
        2. Generate full geocentric h_plus/h_cross waveform.
        3. Project full waveform onto detectors using a geocentric reference time.
        4. Select the usable projected detector-network window.
        5. Choose a common final 4 s output segment containing the windowed network.
        6. Embed the signal into zero-valued 4 s segments for signal-only SNR.
        7. Compute initial SNR.
        8. Optionally rescale distance.
        9. Generate longer processing-context noise segments around the final 4 s segment.
        10. Inject the final windowed projected signals into those context segments.
        11. Process the context segments and crop back to the final 4 s output.
        12. Build labels and metadata.
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

            
            validate_snr_rescaling(
                final_network_snr=final_network.network_snr,
                target_network_snr=snr_decision.target_network_snr,
                relative_tolerance=self.config.snr_relative_tolerance,
            )

        else:
            final_network = initial_network

        processing_length = self.config.processing_length

        context_segment_start_time = (
            final_network.placement.segment_start_time
            - self.config.processing_context_start_seconds
        )

        noises = self.noise_model.sample_network(
            detector_names=self.detector_names,
            seed=int(self.rng.integers(0, 2**32 - 1)),
            length=processing_length,
        )

        noises = {
            detector_name: self.signal_injector.set_strain_start_time(
                strain=noise,
                start_time=context_segment_start_time,
                expected_length=processing_length,
            )
            for detector_name, noise in noises.items()
        }

        # Important: inject the windowed projected network, not the full projection.
        # This injection happens in the longer processing-context segment.
        injected_results = self.signal_injector.inject_network(
            noises=noises,
            signals=final_network.windowed.strains,
        )

        psds = {
            detector_name: self.noise_model.get_psd(
                detector_name,
                length=processing_length,
            )
            for detector_name in self.detector_names
        }

        injected_strains = {
            detector_name: injected_results[detector_name].strain
            for detector_name in self.detector_names
        }

        processed_network = self.signal_processor.process_network(
            strains=injected_strains,
            psds=psds,
        )

        channels: list[np.ndarray] = []

        for detector_name in self.detector_names:
            processed_signal = processed_network[detector_name]

            self._validate_processed_output_alignment(
                processed_signal=processed_signal,
                expected_start_time=final_network.placement.segment_start_time,
                detector_name=detector_name,
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
        progress_every: int = 10,
    ) -> DatasetBatch:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        if max_attempts is None:
            max_attempts = 10 * num_samples

        if progress_every <= 0:
            raise ValueError("progress_every must be positive.")

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        parameters_list: list[CBCParameters] = []
        metadata_list: list[dict] = []

        n_attempts = 0
        n_failed = 0

        start_time = time.perf_counter()

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

            if n_done % progress_every == 0 or n_done == num_samples:
                # Time depuration
                elapsed = time.perf_counter() - start_time
                samples_per_second = n_done / elapsed if elapsed > 0 else float("nan")
                seconds_per_sample = elapsed / n_done if n_done > 0 else float("nan")

                print(
                    f"Built sample {n_done} of {num_samples} "
                    f"--> {n_done / num_samples:.1%} completed "
                    f"(attempts={n_attempts}, failed={n_failed}, "
                    f"elapsed={elapsed:.1f}s, "
                    f"{seconds_per_sample:.2f}s/sample, "
                    f"{samples_per_second:.2f} samples/s)"
                )

        if len(X_list) < num_samples:
            raise RuntimeError(
                f"Could only build {len(X_list)} valid samples out of {num_samples} "
                f"after {n_attempts} attempts. Failed samples: {n_failed}. "
                "Your simulation config is probably rejecting too many samples."
            )

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        # To print the time
        total_elapsed = time.perf_counter() - start_time
        samples_per_second = len(X_list) / total_elapsed if total_elapsed > 0 else float("nan")
        seconds_per_sample = total_elapsed / len(X_list) if len(X_list) > 0 else float("nan")
        failure_rate = n_failed / n_attempts if n_attempts > 0 else 0.0

        print(
            f"--> DATASET GENERATED with {len(X_list)} valid samples "
            f"after {n_attempts} attempts. "
            f"Failed samples: {n_failed} "
            f"({failure_rate:.1%} failure rate). "
            f"Total time: {total_elapsed:.1f}s "
            f"({seconds_per_sample:.2f}s/sample, "
            f"{samples_per_second:.2f} samples/s)."
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

        The full h_plus/h_cross waveform is projected first. The projected
        detector network is then windowed using one common absolute-time window.
        The windowed projected signals are embedded into fixed-duration zero
        segments using the same placement policy later used for actual noise
        injection.
        """
        waveform = self.waveform_generator.generate(params)

        projection = self.detector_projector.project(
            h_plus=waveform.h_plus,
            h_cross=waveform.h_cross,
            parameters=params,
            geocentric_coalescence_time=geocentric_coalescence_time,
        )

        self._validate_detector_set(projection.strains)

        safe_margin_start = float(self.config.safe_margin_start)
        safe_margin_end = float(self.config.safe_margin_end)

        windowed = self.network_window_selector.select(
            projected_strains=projection.strains,
            max_duration=self.config.duration,
        )

        self._validate_detector_set(windowed.strains)

        placement = self.signal_injector.choose_segment_placement_containing_network(
            signals=windowed.strains,
            placement_policy=placement_policy,
            safe_margin_start=safe_margin_start,
            safe_margin_end=safe_margin_end,
            enforce_safe_margins=True,
        )

        zero_segments = {
            detector_name: self.signal_injector.build_zero_strain(
                start_time=placement.segment_start_time,
            )
            for detector_name in windowed.strains
        }

        signal_segment_results = self.signal_injector.inject_network(
            noises=zero_segments,
            signals=windowed.strains,
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
            "initial_parameters": self._parameters_metadata(initial_network.params),
            "final_parameters": self._parameters_metadata(final_network.params),
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
            "processing": self.signal_processor.metadata(),
            "processing_context": self._processing_context_metadata(final_network.placement),
            "labels": self.label_transformer.metadata(),
        }
    
    def _parameters_metadata(self, params: CBCParameters) -> dict:
        return {
            "mass_1": float(params.mass_1),
            "mass_2": float(params.mass_2),
            "distance": float(params.distance),
            "inclination": float(params.inclination),
            "ra": float(params.ra),
            "dec": float(params.dec),
            "spin_1z": float(params.spin_1z),
            "spin_2z": float(params.spin_2z),
            "polarization_angle": float(params.polarization_angle),
            "chirp_mass": float(params.chirp_mass),
            "total_mass": float(params.total_mass),
            "chi_eff": float(params.chi_eff),
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

                "overlap_start_index_strain": int(result.overlap_start_index_strain),
                "overlap_end_index_strain": int(result.overlap_end_index_strain),
                "overlap_start_index_signal": int(result.overlap_start_index_signal),
                "overlap_end_index_signal": int(result.overlap_end_index_signal),

                "n_signal_samples": int(result.n_signal_samples),
                "n_injected_samples": int(result.n_injected_samples),

                "n_clipped_before": int(result.n_clipped_before),
                "n_clipped_after": int(result.n_clipped_after),
                "is_partially_clipped": bool(result.is_partially_clipped),
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

            "processing_context_start_samples": int(self.config.processing_context_start_samples),
            "processing_context_end_samples": int(self.config.processing_context_end_samples),
            "processing_context_start_seconds": float(self.config.processing_context_start_seconds),
            "processing_context_end_seconds": float(self.config.processing_context_end_seconds),
            "processing_length": int(self.config.processing_length),
            "processing_duration": float(self.config.processing_duration),
            "processing_delta_f": float(self.config.processing_delta_f),
            "processing_flength": int(self.config.processing_flength),
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
    
    def _processing_context_metadata(self, placement) -> dict:
        context_start_time = (
            placement.segment_start_time
            - self.config.processing_context_start_seconds
        )
        context_end_time = (
            placement.segment_end_time
            + self.config.processing_context_end_seconds
        )

        return {
            "output_segment_start_time": float(placement.segment_start_time),
            "output_segment_end_time": float(placement.segment_end_time),

            "context_segment_start_time": float(context_start_time),
            "context_segment_end_time": float(context_end_time),

            "output_length": int(self.config.length),
            "output_duration": float(self.config.duration),

            "processing_input_length": int(self.config.processing_length),
            "processing_input_duration": float(self.config.processing_duration),

            "context_start_samples": int(self.config.processing_context_start_samples),
            "context_end_samples": int(self.config.processing_context_end_samples),

            "context_start_seconds": float(self.config.processing_context_start_seconds),
            "context_end_seconds": float(self.config.processing_context_end_seconds),
        }
    

    def _processing_safe_margins_for_internal_padding(self) -> tuple[float, float]:
        """
        Safe margins required to keep the signal away from processing-corrupted edges.

        We take the maximum between config margins and processor-recommended margins.
        """
        processor_start, processor_end = self.signal_processor.recommended_safe_margins()

        safe_margin_start = max(
            float(self.config.safe_margin_start),
            float(processor_start),
        )
        safe_margin_end = max(
            float(self.config.safe_margin_end),
            float(processor_end),
        )

        return safe_margin_start, safe_margin_end


    def _usable_network_duration_for_internal_padding(self) -> float:
        safe_margin_start, safe_margin_end = self._processing_safe_margins_for_internal_padding()

        usable_duration = (
            self.config.duration
            - safe_margin_start
            - safe_margin_end
        )

        if usable_duration <= 0:
            raise ValueError(
                "Safe margins leave no usable duration for the signal. "
                f"duration={self.config.duration}, "
                f"safe_margin_start={safe_margin_start}, "
                f"safe_margin_end={safe_margin_end}."
            )

        return float(usable_duration)

    def _validate_detector_set(self, signals: dict[str, TimeSeries]) -> None:
        signal_detectors = set(signals.keys())
        expected_detectors = set(self.detector_names)

        if signal_detectors != expected_detectors:
            raise ValueError(
                "Projected signal detector set does not match builder detector_names. "
                f"signals={signal_detectors}, expected={expected_detectors}"
            )

    def _validate_processed_output_alignment(
        self,
        processed_signal: TimeSeries,
        expected_start_time: float,
        detector_name: str,
    ) -> None:
        if len(processed_signal) != self.config.length:
            raise ValueError(
                "Processed output length mismatch. "
                f"detector={detector_name}, "
                f"got={len(processed_signal)}, expected={self.config.length}."
            )

        if processed_signal.delta_t != self.config.delta_t:
            raise ValueError(
                "Processed output delta_t mismatch. "
                f"detector={detector_name}, "
                f"got={processed_signal.delta_t}, expected={self.config.delta_t}."
            )

        dt_error = abs(
            float(processed_signal.start_time)
            - float(expected_start_time)
        )

        if dt_error > self.config.delta_t:
            raise ValueError(
                "Processed output start_time is not aligned with final placement. "
                f"detector={detector_name}, "
                f"processed_start_time={float(processed_signal.start_time)}, "
                f"expected_start_time={expected_start_time}, "
                f"dt_error={dt_error}, "
                f"delta_t={self.config.delta_t}."
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
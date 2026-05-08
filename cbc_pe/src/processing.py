import numpy as np

from pycbc.types.timeseries import TimeSeries

from .config import SimulationConfig


class SignalProcessor:
    def __init__(
        self,
        config: SimulationConfig,
        apply_whitening: bool = False,
        apply_lowpass: bool = False,
        apply_highpass: bool = False,
        apply_standardization: bool = False,
        preserve_length: bool = True,
        lowpass_frequency: float = 350.0,
        highpass_frequency: float = 30.0,
        fir_order: int = 512,
        fir_beta: float = 0.5,
        whitening_max_filter_fraction: float = 0.25,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.apply_whitening = apply_whitening
        self.apply_lowpass = apply_lowpass
        self.apply_highpass = apply_highpass
        self.apply_standardization = apply_standardization
        self.preserve_length = preserve_length

        self.lowpass_frequency = lowpass_frequency
        self.highpass_frequency = highpass_frequency

        self.fir_order = fir_order
        self.fir_beta = fir_beta
        self.whitening_max_filter_fraction = whitening_max_filter_fraction

        self.rng = rng if rng is not None else np.random.default_rng()

        self._validate_config()

    def process(self, strain: TimeSeries) -> TimeSeries:
        """
        Process one detector strain segment.

        The input is expected to be a fixed-duration TimeSeries with:
            len(strain) == config.length
            strain.delta_t == config.delta_t

        If preserve_length=True, the output is padded back to the original
        length after operations that remove corrupted edge samples.
        """
        self._validate_input(strain)

        processing_strain = strain.copy()

        original_length = len(processing_strain)
        original_start_time = float(processing_strain.start_time)

        if self.apply_whitening:
            processing_strain = processing_strain.whiten(
                segment_duration=processing_strain.get_duration(),
                max_filter_duration=(
                    processing_strain.get_duration()
                    * self.whitening_max_filter_fraction
                ),
                trunc_method="hann",
                remove_corrupted=True,
                low_frequency_cutoff=self.highpass_frequency,
                return_psd=False,
            )

        if self.apply_lowpass:
            processing_strain = processing_strain.lowpass_fir(
                self.lowpass_frequency,
                order=self.fir_order,
                beta=self.fir_beta,
                remove_corrupted=True,
            )

        if self.apply_highpass:
            processing_strain = processing_strain.highpass_fir(
                self.highpass_frequency,
                order=self.fir_order,
                beta=self.fir_beta,
                remove_corrupted=True,
            )

        if self.apply_standardization:
            processing_strain = self._standardize(processing_strain)

        if self.preserve_length:
            processing_strain = self._restore_length(
                processing_strain=processing_strain,
                original_length=original_length,
                original_start_time=original_start_time,
            )

        self._validate_output(processing_strain)

        return processing_strain
    
    def process_network(self, strains: dict[str, TimeSeries]) -> dict[str, TimeSeries]:
        processed = {}

        for detector, strain in strains.items():
            processed[detector] = self.process(strain)

        return processed

    def metadata(self) -> dict:
        return {
            "apply_whitening": self.apply_whitening,
            "apply_lowpass": self.apply_lowpass,
            "apply_highpass": self.apply_highpass,
            "apply_standardization": self.apply_standardization,
            "preserve_length": self.preserve_length,
            "lowpass_frequency": self.lowpass_frequency,
            "highpass_frequency": self.highpass_frequency,
            "fir_order": self.fir_order,
            "fir_beta": self.fir_beta,
            "whitening_max_filter_fraction": self.whitening_max_filter_fraction,
        }

    def _standardize(self, strain: TimeSeries) -> TimeSeries:
        delta_t = strain.delta_t
        epoch = strain.start_time

        array = np.asarray(strain)

        mean = float(np.mean(array))
        std = float(np.std(array))

        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("Cannot standardize strain with non-finite mean/std.")

        if std == 0.0:
            standardized = np.zeros_like(array)
        else:
            standardized = (array - mean) / std

        return TimeSeries(
            initial_array=standardized,
            delta_t=delta_t,
            epoch=epoch,
        )

    def _restore_length(
        self,
        processing_strain: TimeSeries,
        original_length: int,
        original_start_time: float,
    ) -> TimeSeries:
        new_length = len(processing_strain)
        diff_length = original_length - new_length

        if diff_length < 0:
            raise ValueError(
                "Processed strain is longer than the original strain. "
                f"processed_length={new_length}, original_length={original_length}."
            )

        if diff_length == 0:
            processing_strain.start_time = original_start_time
            return processing_strain

        left_pad = diff_length // 2
        right_pad = diff_length - left_pad

        processing_strain.prepend_zeros(left_pad)
        processing_strain.append_zeros(right_pad)

        processing_strain.start_time = original_start_time

        return processing_strain

    def _validate_config(self) -> None:
        if self.lowpass_frequency <= 0:
            raise ValueError("lowpass_frequency must be positive.")

        if self.highpass_frequency <= 0:
            raise ValueError("highpass_frequency must be positive.")

        if self.highpass_frequency >= self.lowpass_frequency:
            raise ValueError(
                "highpass_frequency must be smaller than lowpass_frequency."
            )

        nyquist = 0.5 * self.config.sampling_frequency

        if self.lowpass_frequency >= nyquist:
            raise ValueError(
                "lowpass_frequency must be smaller than Nyquist frequency. "
                f"lowpass_frequency={self.lowpass_frequency}, nyquist={nyquist}."
            )

        if self.fir_order <= 0:
            raise ValueError("fir_order must be positive.")

        if self.fir_beta <= 0:
            raise ValueError("fir_beta must be positive.")

        if not (0.0 < self.whitening_max_filter_fraction < 1.0):
            raise ValueError(
                "whitening_max_filter_fraction must be in (0, 1)."
            )

        if not self.preserve_length:
            raise NotImplementedError(
                "preserve_length=False is not supported yet."
            )

    def _validate_input(self, strain: TimeSeries) -> None:
        if not isinstance(strain, TimeSeries):
            raise TypeError("strain must be a TimeSeries.")

        if len(strain) != self.config.length:
            raise ValueError(
                f"Input strain length mismatch: got {len(strain)}, "
                f"expected {self.config.length}."
            )

        if strain.delta_t != self.config.delta_t:
            raise ValueError(
                f"Input strain delta_t mismatch: got {strain.delta_t}, "
                f"expected {self.config.delta_t}."
            )

        if not np.all(np.isfinite(strain.numpy())):
            raise ValueError("Input strain contains NaN or Inf.")

    def _validate_output(self, strain: TimeSeries) -> None:
        if len(strain) != self.config.length:
            raise ValueError(
                f"Output strain length mismatch: got {len(strain)}, "
                f"expected {self.config.length}."
            )

        if strain.delta_t != self.config.delta_t:
            raise ValueError(
                f"Output strain delta_t mismatch: got {strain.delta_t}, "
                f"expected {self.config.delta_t}."
            )

        if not np.all(np.isfinite(strain.numpy())):
            raise ValueError("Output strain contains NaN or Inf.")
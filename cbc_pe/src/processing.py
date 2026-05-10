from __future__ import annotations

import numpy as np
import math
import traceback

from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.psd import inverse_spectrum_truncation

from .config import SimulationConfig


class SignalProcessor:
    def __init__(
        self,
        config: SimulationConfig,
        whitening_method: str = "psd",  # "none", "pycbc_local"(strain: noise + signal), "psd" (default, from noise calculation)
        apply_lowpass: bool = False,
        apply_highpass: bool = False,
        apply_standardization: bool = False,
        output_mode: str = "crop_to_config", # "restore_length" -> padding to original lengtg and "crop_to_config" -> output with config.length (uses preserve_length=False)
        lowpass_frequency: float = 512.0,
        highpass_frequency: float = 30.0,
        whitening_low_frequency_cutoff: float | None = None,
        whitening_max_filter_duration: float = 0.5,
        whitening_trunc_method: str | None = "hann",
        fir_order: int = 256,
        fir_beta: float = 5.0,
        remove_corrupted: bool = True,
        rng: np.random.Generator | None = None, 
    ) -> None:
        self.config = config

        self.whitening_method = whitening_method
        self.apply_lowpass = apply_lowpass
        self.apply_highpass = apply_highpass
        self.apply_standardization = apply_standardization
        self.output_mode = output_mode

        self.lowpass_frequency = lowpass_frequency
        self.highpass_frequency = highpass_frequency

        if whitening_low_frequency_cutoff is None:
            whitening_low_frequency_cutoff = highpass_frequency

        self.whitening_low_frequency_cutoff = whitening_low_frequency_cutoff
        self.whitening_max_filter_duration = whitening_max_filter_duration
        self.whitening_trunc_method = whitening_trunc_method

        self.fir_order = fir_order
        self.fir_beta = fir_beta
        self.remove_corrupted = remove_corrupted

        self.rng = rng if rng is not None else np.random.default_rng()

        self._validate_config()

    def process(
        self,
        strain: TimeSeries,
        psd: FrequencySeries | None = None,
        detector_name: str | None = None,
    ) -> TimeSeries:
        """
        Process one detector strain segment.

        If whitening_method == "psd", a detector-specific PSD must be passed.
        """
        self._validate_input(strain)

        processing_strain = TimeSeries(strain.copy())

        original_length = len(processing_strain)
        original_start_time = float(processing_strain.start_time)

        if self.whitening_method == "pycbc_local":
            processing_strain = self._whiten_pycbc_local(processing_strain)

        elif self.whitening_method == "psd":
            if psd is None:
                traceback.print_stack(limit=8)
                raise ValueError(
                    "psd must be provided when whitening_method='psd'. "
                    f"detector_name={detector_name}"
                )

            processing_strain = self._whiten_with_psd(
                strain=processing_strain,
                psd=psd,
            )

        elif self.whitening_method == "none":
            pass

        else:
            raise ValueError(f"Unknown whitening_method: {self.whitening_method}")

        if self.apply_highpass:
            processing_strain = processing_strain.highpass_fir(
                self.highpass_frequency,
                order=self.fir_order,
                beta=self.fir_beta,
                remove_corrupted=self.remove_corrupted,
            )

        if self.apply_lowpass:
            processing_strain = processing_strain.lowpass_fir(
                self.lowpass_frequency,
                order=self.fir_order,
                beta=self.fir_beta,
                remove_corrupted=self.remove_corrupted,
            )

        if self.apply_standardization:
            processing_strain = self._standardize(processing_strain)

        if self.output_mode == "restore_length":
            processing_strain = self._restore_length(
                processing_strain=processing_strain,
                original_length=original_length,
                original_start_time=original_start_time,
            )

        elif self.output_mode == "crop_to_config":
            processing_strain = self._crop_to_config_output(
                processing_strain=processing_strain,
                input_start_time=original_start_time,
            )

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        self._validate_output(processing_strain)

        return processing_strain

    def process_network(
        self,
        strains: dict[str, TimeSeries],
        psds: dict[str, FrequencySeries] | None = None,
    ) -> dict[str, TimeSeries]:

        if self.whitening_method == "psd" and psds is None:
            raise ValueError(
                "psds must be provided when whitening_method='psd'."
            )

        processed = {}

        for detector_name, strain in strains.items():
            psd = None if psds is None else psds[detector_name]

            processed[detector_name] = self.process(
                strain=strain,
                psd=psd,
                detector_name=detector_name,
            )

        return processed


    def _crop_to_config_output(
        self,
        processing_strain: TimeSeries,
        input_start_time: float,
    ) -> TimeSeries:
        """
        Crop a processed context segment to the final fixed-duration model window.

        The input context segment is assumed to start at:
            final_output_start_time - config.processing_context_start_seconds

        Therefore the returned output should start at:
            input_start_time + config.processing_context_start_seconds

        and have length config.length.
        """
        target_start_time = (
            input_start_time
            + self.config.processing_context_start_seconds
        )

        start_index = int(round(
            (target_start_time - float(processing_strain.start_time))
            / self.config.delta_t
        ))
        end_index = start_index + self.config.length

        if start_index < 0 or end_index > len(processing_strain):
            raise ValueError(
                "Cannot crop processed strain to requested output window. "
                f"processed_start_time={float(processing_strain.start_time)}, "
                f"processed_len={len(processing_strain)}, "
                f"input_start_time={input_start_time}, "
                f"target_start_time={target_start_time}, "
                f"start_index={start_index}, "
                f"end_index={end_index}, "
                f"target_length={self.config.length}."
            )

        cropped = processing_strain[start_index:end_index]
        cropped.start_time = target_start_time

        return cropped

    def _whiten_pycbc_local(self, strain: TimeSeries) -> TimeSeries:
        return strain.whiten(
            segment_duration=strain.get_duration(),
            max_filter_duration=self.whitening_max_filter_duration,
            trunc_method=self.whitening_trunc_method,
            remove_corrupted=self.remove_corrupted,
            low_frequency_cutoff=self.whitening_low_frequency_cutoff,
            return_psd=False,
        )

    def _whiten_with_psd(
        self,
        strain: TimeSeries,
        psd: FrequencySeries,
    ) -> TimeSeries:
        self._validate_psd(strain=strain, psd=psd)

        max_filter_len = int(round(
            self.whitening_max_filter_duration * strain.sample_rate
        ))

        conditioned_psd = inverse_spectrum_truncation(
            psd.copy(),
            max_filter_len=max_filter_len,
            low_frequency_cutoff=self.whitening_low_frequency_cutoff,
            trunc_method=self.whitening_trunc_method,
        )

        white = (
            strain.to_frequencyseries() / conditioned_psd**0.5
        ).to_timeseries()

        white.start_time = strain.start_time

        if self.remove_corrupted:
            left = max_filter_len // 2
            right = max_filter_len - left

            if left + right >= len(white):
                raise ValueError(
                    "Whitening corruption length is too large for the segment. "
                    f"max_filter_len={max_filter_len}, len={len(white)}"
                )

            white = white[left:len(white) - right]

        return white

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

    def metadata(self) -> dict:
        return {
            "whitening_method": self.whitening_method,
            "apply_lowpass": self.apply_lowpass,
            "apply_highpass": self.apply_highpass,
            "apply_standardization": self.apply_standardization,
            "lowpass_frequency": float(self.lowpass_frequency),
            "highpass_frequency": float(self.highpass_frequency),
            "whitening_low_frequency_cutoff": (
                None
                if self.whitening_low_frequency_cutoff is None
                else float(self.whitening_low_frequency_cutoff)
            ),
            "whitening_max_filter_duration": float(self.whitening_max_filter_duration),
            "whitening_trunc_method": self.whitening_trunc_method,
            "fir_order": int(self.fir_order),
            "fir_beta": float(self.fir_beta),
            "remove_corrupted": bool(self.remove_corrupted),
            "processing_preset": self.processing_preset_name(),

            "corrupted_margin_samples_per_side": int(self.corrupted_margin_samples_per_side()),
            "corrupted_margin_seconds_per_side": float(self.corrupted_margin_seconds_per_side()),
            "recommended_safe_margin_start": float(self.recommended_safe_margins()[0]),
            "recommended_safe_margin_end": float(self.recommended_safe_margins()[1]),
            "usable_duration_after_processing_margins": float(self.usable_duration_after_processing_margins()),

            "uses_processing_context": bool(self.output_mode == "crop_to_config"),
            "output_mode": self.output_mode,
            "output_length": int(self.config.length),
            "output_duration": float(self.config.duration),
            "processing_input_length": int(self.config.processing_length),
            "processing_input_duration": float(self.config.processing_duration),
            "processing_context_start_samples": int(self.config.processing_context_start_samples),
            "processing_context_end_samples": int(self.config.processing_context_end_samples),
            "processing_context_start_seconds": float(self.config.processing_context_start_seconds),
            "processing_context_end_seconds": float(self.config.processing_context_end_seconds),
        }

    def processing_preset_name(self) -> str:
        if self.whitening_method == "none" and not self.apply_highpass and not self.apply_lowpass:
            return "raw_identity"

        parts = [self.whitening_method]

        if self.apply_highpass:
            parts.append(f"hp{self.highpass_frequency:g}")

        if self.apply_lowpass:
            parts.append(f"lp{self.lowpass_frequency:g}")

        parts.append(f"fir{self.fir_order}")
        parts.append(f"beta{self.fir_beta:g}")
        parts.append(self.output_mode)

        if self.remove_corrupted:
            parts.append("crop")
        else:
            parts.append("nocrop")

        return "_".join(parts)



    def corrupted_margin_samples_per_side(self) -> int:
        """
        Estimate how many samples at each edge are unreliable/corrupted by
        whitening and FIR filtering.

        This is used to force signal placement away from corrupted/padded edges.
        """
        samples = 0

        # Whitening edge corruption.
        if self.whitening_method in {"pycbc_local", "psd"}:
            max_filter_len = int(round(
                self.whitening_max_filter_duration * self.config.sampling_frequency
            ))

            samples += math.ceil(max_filter_len / 2)

        # FIR edge corruption. In PyCBC FIR filters, order is effectively the
        # number of corrupted samples per side.
        if self.apply_highpass:
            samples += int(self.fir_order)

        if self.apply_lowpass:
            samples += int(self.fir_order)

        return samples


    def corrupted_margin_seconds_per_side(self) -> float:
        return self.corrupted_margin_samples_per_side() * self.config.delta_t


    def recommended_safe_margins(self, extra_margin_seconds: float = 0.025) -> tuple[float, float]:
        """
        Return recommended start/end safe margins in seconds.

        extra_margin_seconds is a small buffer for rounding and implementation
        differences.
        """
        margin = self.corrupted_margin_seconds_per_side() + extra_margin_seconds
        return float(margin), float(margin)


    def usable_duration_after_processing_margins(
        self,
        safe_margin_start: float | None = None,
        safe_margin_end: float | None = None,
    ) -> float:
        if safe_margin_start is None or safe_margin_end is None:
            safe_margin_start, safe_margin_end = self.recommended_safe_margins()

        usable_duration = (
            self.config.duration
            - safe_margin_start
            - safe_margin_end
        )

        if usable_duration <= 0:
            raise ValueError(
                "Processing safe margins are too large for the segment duration. "
                f"duration={self.config.duration}, "
                f"safe_margin_start={safe_margin_start}, "
                f"safe_margin_end={safe_margin_end}."
            )

        return float(usable_duration)



    def _validate_config(self) -> None:
        allowed_methods = {"none", "pycbc_local", "psd"}

        if self.whitening_method not in allowed_methods:
            raise ValueError(
                f"whitening_method must be one of {allowed_methods}, "
                f"got {self.whitening_method}"
            )

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

        if self.whitening_low_frequency_cutoff is not None:
            if self.whitening_low_frequency_cutoff < 0:
                raise ValueError("whitening_low_frequency_cutoff must be non-negative.")

            if self.whitening_low_frequency_cutoff >= nyquist:
                raise ValueError(
                    "whitening_low_frequency_cutoff must be smaller than Nyquist."
                )

        if self.whitening_max_filter_duration <= 0:
            raise ValueError("whitening_max_filter_duration must be positive.")

        if self.whitening_max_filter_duration >= self.config.duration:
            raise ValueError(
                "whitening_max_filter_duration must be smaller than segment duration."
            )

        if self.fir_order <= 0:
            raise ValueError("fir_order must be positive.")

        if self.fir_beta <= 0:
            raise ValueError("fir_beta must be positive.")


        allowed_output_modes = {"restore_length", "crop_to_config"}

        if self.output_mode not in allowed_output_modes:
            raise ValueError(
                f"output_mode must be one of {allowed_output_modes}, "
                f"got {self.output_mode}"
            )

        if self.output_mode == "crop_to_config":
            if self.config.processing_length <= self.config.length:
                raise ValueError(
                    "output_mode='crop_to_config' requires processing_length "
                    "larger than config.length. "
                    f"processing_length={self.config.processing_length}, "
                    f"length={self.config.length}."
                )


    def _expected_input_length(self) -> int:
        if self.output_mode == "crop_to_config":
            return self.config.processing_length

        return self.config.length

    def _validate_input(self, strain: TimeSeries) -> None:
        if not isinstance(strain, TimeSeries):
            raise TypeError("strain must be a TimeSeries.")

        expected_length = self._expected_input_length()

        if len(strain) != expected_length:
            raise ValueError(
                f"Input strain length mismatch: got {len(strain)}, "
                f"expected {expected_length} for output_mode={self.output_mode}."
            )

        if strain.delta_t != self.config.delta_t:
            raise ValueError(
                f"Input strain delta_t mismatch: got {strain.delta_t}, "
                f"expected {self.config.delta_t}."
            )

        if not np.all(np.isfinite(strain.numpy())):
            raise ValueError("Input strain contains NaN or Inf.")

    def _validate_psd(
        self,
        strain: TimeSeries,
        psd: FrequencySeries,
    ) -> None:
        if not isinstance(psd, FrequencySeries):
            raise TypeError("psd must be a pycbc.types.frequencyseries.FrequencySeries.")

        strain_frequency = strain.to_frequencyseries()

        expected_len = len(strain_frequency)

        if len(psd) != expected_len:
            raise ValueError(
                "PSD length mismatch. "
                f"got len(psd)={len(psd)}, expected {expected_len}"
            )

        if abs(float(psd.delta_f) - float(strain_frequency.delta_f)) > 0.0:
            raise ValueError(
                "PSD delta_f mismatch. "
                f"got {psd.delta_f}, expected {strain_frequency.delta_f}"
            )

        psd_array = psd.numpy()

        if not np.all(np.isfinite(psd_array)):
            raise ValueError("PSD contains NaN or Inf.")

        if np.any(psd_array < 0):
            raise ValueError("PSD contains negative values.")

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
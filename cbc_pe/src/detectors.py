from __future__ import annotations
from dataclasses import dataclass

from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries

from .parameters import CBCParameters


@dataclass(frozen=True)
class ProjectionMetadata:
    """
    Some definitions:
    -------------------------------------------------------------------------------------------
    geocentric_coalescence_time
        Geocentric reference time 

    delay
        How advanced/delayed is the detector wrt the geocenter. 

    arrival_time
        Absolute time when t=0 of the signal arrives the detector

    projected_start_time
        Absolute time of the first point(sample) of the signal proyected onto that detector

    projected_end_time
        Absolute time just after the last sample of the proyected signal.
    -------------------------------------------------------------------------------------------
    """
    detector_names: list[str]
    geocentric_coalescence_time: float
    expected_detector_time_delays: dict[str, float]
    detector_arrival_times: dict[str, float] # This is the physical time where the coalescence signal arrives the detector (peak)
    projected_start_times: dict[str, float] # when the signal starts at detector
    projected_end_times: dict[str, float] # when the signal ends


@dataclass(frozen=True)
class ProjectedStrains:
    strains: dict[str, TimeSeries]
    metadata: ProjectionMetadata


class DetectorProjector:
    def __init__(self, detector_names: list[str] | None = None):
        """
        Project source-frame plus/cross polarizations onto detector strain.

        Parameters
        ----------
        detector_names : list[str] | None
            Detector names. If None, use H1, L1, V1.
        """
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        if len(detector_names) == 0:
            raise ValueError("detector_names cannot be empty.")

        self.detectors = {name: Detector(name) for name in detector_names}

    def project(
        self,
        h_plus: TimeSeries,
        h_cross: TimeSeries,
        parameters: CBCParameters,
        geocentric_coalescence_time: float,
    ) -> ProjectedStrains:
        """
        Project h_plus/h_cross onto each detector.

        The input waveform is assumed to use t=0 as the geocentric
        coalescence/reference time. Before projection, the waveform epoch is
        shifted to an absolute GPS-like geocentric coalescence time.

        Detector.project_wave(..., method="lal") is then responsible for
        applying the detector antenna response and physical time delay.
        """
        self._validate_waveform_pair(h_plus, h_cross)

        h_plus_abs = h_plus.copy()
        h_cross_abs = h_cross.copy()

        relative_start_time = float(h_plus.start_time)
        absolute_start_time = geocentric_coalescence_time + relative_start_time

        h_plus_abs.start_time = absolute_start_time
        h_cross_abs.start_time = absolute_start_time

        expected_detector_time_delays = self.compute_time_delays(
            parameters=parameters,
            geocentric_coalescence_time=geocentric_coalescence_time,
        )

        detector_arrival_times = {
            name: geocentric_coalescence_time + delay
            for name, delay in expected_detector_time_delays.items()
        }

        strains: dict[str, TimeSeries] = {}

        for name, detector in self.detectors.items():
            strain = detector.project_wave(
                h_plus_abs,
                h_cross_abs,
                parameters.ra,
                parameters.dec,
                parameters.polarization_angle,
                method="lal",
            )
            strains[name] = strain

        projected_start_times = {
            name: float(strain.start_time)
            for name, strain in strains.items()
        }

        projected_end_times = {
            name: float(strain.start_time) + len(strain) * float(strain.delta_t)
            for name, strain in strains.items()
        }

        metadata = ProjectionMetadata(
            detector_names=list(self.detectors.keys()),
            geocentric_coalescence_time=geocentric_coalescence_time,
            expected_detector_time_delays=expected_detector_time_delays,
            detector_arrival_times=detector_arrival_times,
            projected_start_times=projected_start_times,
            projected_end_times=projected_end_times,
        )

        return ProjectedStrains(
            strains=strains,
            metadata=metadata,
        )

    def compute_time_delays(
        self,
        parameters: CBCParameters,
        geocentric_coalescence_time: float,
    ) -> dict[str, float]:
        """
        Compute detector time delays relative to the Earth center.

        This method is used for metadata and validation only. The actual
        time shifting of projected waveforms is handled by Detector.project_wave
        when project(..., method="lal") is used.
    
        Parameters
        ----------
        parameters : CBCParameters
            Source sky location.
        geocentric_coalescence_time : float
            Absolute geocentric GPS-like coalescence/reference time.

        Returns
        -------
        dict[str, float]
            Detector delay relative to the geocenter, in seconds.
        """
        time_delays: dict[str, float] = {}

        for name, detector in self.detectors.items():
            time_delay = detector.time_delay_from_earth_center(
                right_ascension=parameters.ra,
                declination=parameters.dec,
                t_gps=geocentric_coalescence_time,
            )
            time_delays[name] = float(time_delay)

        return time_delays

    @staticmethod
    def _validate_waveform_pair(
        h_plus: TimeSeries,
        h_cross: TimeSeries,
    ) -> None:
        if len(h_plus) != len(h_cross):
            raise ValueError("h_plus and h_cross must have the same length.")

        if h_plus.delta_t != h_cross.delta_t:
            raise ValueError("h_plus and h_cross must have the same delta_t.")

        if h_plus.start_time != h_cross.start_time:
            raise ValueError("h_plus and h_cross must have the same start_time.")

        if len(h_plus) == 0:
            raise ValueError("Waveform cannot be empty.")
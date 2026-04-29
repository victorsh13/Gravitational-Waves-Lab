import numpy as np
from .parameters import CBCParameters
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries

class DetectorProjector:
    def __init__(self, detector_names: list[str] | None = None):
        """
        Initialize a DetectorProjector object.

        Parameters
        ----------
        detector_names : list[str] | None
            List of detector names to be used. If None, use all three detectors: H1, L1, and V1.
            

        Notes
        -----
        DetectorProjector projects the strain onto the sky plane for each detector.


        """
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        self.detectors = {name: Detector(name) for name in detector_names}

    def project(self, h_plus: TimeSeries, h_cross: TimeSeries, parameters: CBCParameters) -> dict[str, TimeSeries]:
        """
        Project the strain onto the sky plane for each detector.

        Parameters
        ----------
        h_plus : TimeSeries
            The strain in the plus direction.
        h_cross : TimeSeries
            The strain in the cross direction.
        parameters : CBCParameters
            The parameters of the binary compact object.

        Returns
        -------
        dict[str, float]    
            A dictionary containing the projected strain for each detector.
        """    


        strains = {}

        for name, detector in self.detectors.items():
            strain = detector.project_wave(
                h_plus,
                h_cross,
                parameters.ra,
                parameters.dec,
                parameters.polarization_angle,
                method="lal",
            )
            strains[name] = strain

        return strains

    def compute_time_delays(
            self, 
            parameters: CBCParameters, 
            geo_injection_time: float,
            ) -> dict[str, float]:
        """ 
        Compute the time delays for each detector, given a geocentric injection time for the signal.

        WARNING: This method a relative time within the signal segment, not the absolute time. 

        Parameters
        ----------
        parameters : CBCParameters
            The parameters of the binary compact object. Right ascension and declination are used to compute the time delays.
        geo_injection_time : float
            The geocentric injection time for the signal.

        Returns
        -------
        dict[str, float]
            A dictionary containing the time delays for each detector.
        """
        time_delays = {}
        for name, detector in self.detectors.items():
            time_delay = detector.time_delay_from_earth_center( 
                        right_ascension=parameters.ra,
                        declination=parameters.dec, 
                        t_gps=geo_injection_time)
            time_delays[name] = time_delay

        return time_delays

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

    def project(self, h_plus: TimeSeries, h_cross: TimeSeries, parameters: CBCParameters) -> dict[str, float]:
        strains = {}

        for name, detector in self.detectors.items():
            strain = detector.project_wave(
                h_plus,
                h_cross,
                parameters.ra,
                parameters.dec,
                1,
                method="lal",
            )
            strains[name] = strain

        return strains

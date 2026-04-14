from .parameters import CBCParameters
from pycbc.detector import Detector
import pycbc.detector as pycbc_det

class DetectorProjector:
    def __init__(self, detector_names: list[str] | None = None):
        if detector_names is None:
            detector_names = ["H1", "L1", "V1"]

        self.detectors = {
            name: Detector(name) for name in detector_names
        }

        print("Available detectors:", pycbc_det.get_available_detectors())

    def project(self, h_plus, h_cross, parameters: CBCParameters) -> dict[str, float]:
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

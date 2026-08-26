import numpy as np

from src.config import SimulationConfig
from src.dataset import DatasetBuilder


DETECTORS = ["H1", "L1", "V1"]


def make_builder(seed=123):
    config = SimulationConfig(
        sampling_frequency=4096.0,
        duration=4.0,
        low_frequency_cutoff=30.0,
        waveform_approximant="SEOBNRv4_opt",
        target_network_snr_range=None,
        processing_context_start_samples=1664,
        processing_context_end_samples=1664,
    )

    return DatasetBuilder.from_config(
        config=config,
        detector_names=DETECTORS,
        signal_processor_kwargs={
            "whitening_method": "psd",
            "output_mode": "crop_to_config",
        },
        rng=np.random.default_rng(seed),
    )


def test_m10_windowed_context_output_contract():
    builder = make_builder()

    sample = builder.build_sample(
        strain_mode="gw_only",
        signal_context_mode="m10_windowed",
        geocentric_coalescence_time=1126259462.0,
    )

    assert sample.X.shape == (
        len(DETECTORS),
        builder.config.length,
    )

    assert np.all(np.isfinite(sample.X))


from src.parameters import CBCParameters


def low_mass_params():
    return CBCParameters(
        mass_1=5.0,
        mass_2=5.0,
        distance=500.0,
        inclination=1.0,
        ra=1.0,
        dec=0.3,
        spin_1z=0.0,
        spin_2z=0.0,
        polarization_angle=0.5,
    )


def test_full_projection_injects_more_physical_signal_than_m10_window():
    params = low_mass_params()

    builder_a = make_builder(seed=123)
    builder_b = make_builder(seed=123)

    sample_a = builder_a.build_sample(
        params=params,
        strain_mode="gw_only",
        signal_context_mode="m10_windowed",
        geocentric_coalescence_time=1126259462.0,
    )

    sample_b = builder_b.build_sample(
        params=params,
        strain_mode="gw_only",
        signal_context_mode="full_projection",
        geocentric_coalescence_time=1126259462.0,
    )

    assert sample_b.metadata["windowing"]["is_truncated"]

    inj_a = sample_a.metadata["injection"]
    inj_b = sample_b.metadata["injection"]
    
    for ifo in DETECTORS:
        assert (
            inj_b[ifo]["n_injected_samples"]
            >
            inj_a[ifo]["n_injected_samples"]
        )


def test_context_mode_does_not_change_final_output_contract():
    params = low_mass_params()

    outputs = {}

    for mode in ["m10_windowed", "full_projection"]:
        builder = make_builder(seed=123)

        sample = builder.build_sample(
            params=params,
            strain_mode="gw_only",
            signal_context_mode=mode,
            geocentric_coalescence_time=1126259462.0,
        )

        outputs[mode] = sample

    assert outputs["m10_windowed"].X.shape == outputs["full_projection"].X.shape
    assert outputs["m10_windowed"].X.shape[1] == 16384

    p_a = outputs["m10_windowed"].metadata["processing_context"]
    p_b = outputs["full_projection"].metadata["processing_context"]

    assert p_a["output_segment_start_time"] == p_b["output_segment_start_time"]
    assert p_a["output_segment_end_time"] == p_b["output_segment_end_time"]


def test_context_mode_does_not_change_projection_geometry():
    params = low_mass_params()

    samples = {}

    for mode in ["m10_windowed", "full_projection"]:
        builder = make_builder(seed=123)

        samples[mode] = builder.build_sample(
            params=params,
            strain_mode="gw_only",
            signal_context_mode=mode,
            geocentric_coalescence_time=1126259462.0,
        )

    proj_a = samples["m10_windowed"].metadata["projection"]
    proj_b = samples["full_projection"].metadata["projection"]

    assert (
        proj_a["expected_detector_time_delays"]
        == proj_b["expected_detector_time_delays"]
    )

    assert (
        proj_a["detector_arrival_times"]
        == proj_b["detector_arrival_times"]
    )
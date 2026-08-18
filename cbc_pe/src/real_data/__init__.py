"""
Utilities for inference and evaluation on real gravitational-wave data.

Modules
-------
gwosc_utils
    Low-level GWOSC strain-file operations:
    HDF5 validation, download/cache handling, TimeSeries loading,
    temporal bounds and finite-data checks.

catalog
    GWOSC catalog and event-metadata operations:
    catalog API access, parameter flattening, event tables,
    detector availability and strain-file URL resolution.

psd
    Off-source PSD handling:
    PSD-window validation and selection, and PSD estimation.

signal_processing
    Real-strain processing for model input:
    event-window validation, extraction of detector segments and
    application of the production SignalProcessor.

inference
    Neural-network inference on processed real inputs:
    standardized predictions, physical-unit predictions and embeddings.

lvk_reference
    Construction of LVK reference quantities:
    source-frame to detector-frame mass conversion and uncertainty
    propagation for comparison with model predictions.

Design principle
----------------
This package contains reusable real-data infrastructure. Model-specific
scientific orchestration, experiment policy and presentation logic should
remain outside these modules unless they form a stable reusable interface.
"""


"""
Only the part that depends on features and kNN:

extracting difficulty scores
computing sigmas
maybe a class type KNNDifficultyEstimator

This would separate because it depends on model/features, and I don't want to contaminate the basic binning with TensorFlow or embeddings.
"""

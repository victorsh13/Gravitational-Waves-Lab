import unittest

import numpy as np
import torch

from src.real_data.inference import (
    predict_real_with_embeddings,
)


class DummyEmbeddingRegressor(torch.nn.Module):
    """
    Deterministic toy model implementing the same inference interface
    required by extract_predictions_and_embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        return_embedding=False,
    ):
        # x: (B, C, T)

        # Deterministic 3D embedding.
        emb = x.mean(dim=2)

        # Deterministic 3-label prediction.
        pred = torch.stack(
            [
                emb[:, 0] + emb[:, 1],
                emb[:, 1] - emb[:, 2],
                0.5 * emb[:, 0],
            ],
            dim=1,
        )

        if return_embedding:
            return pred, emb

        return pred


class TestRealDataInference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)

        self.X = rng.normal(
            size=(7, 3, 64),
        ).astype(np.float32)

        self.y_mean = np.array(
            [30.0, 70.0, 0.05],
            dtype=np.float32,
        )

        self.y_std = np.array(
            [10.0, 20.0, 0.25],
            dtype=np.float32,
        )

        self.model = DummyEmbeddingRegressor()
        self.device = torch.device("cpu")

    def test_predictions_match_direct_model_output(self):
        pred_std, pred_phys, emb = (
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X,
                device=self.device,
                y_mean=self.y_mean,
                y_std=self.y_std,
                batch_size=3,
            )
        )

        with torch.no_grad():
            expected_pred, expected_emb = (
                self.model(
                    torch.from_numpy(self.X),
                    return_embedding=True,
                )
            )

        expected_pred = (
            expected_pred.numpy()
        )

        expected_emb = (
            expected_emb.numpy()
        )

        np.testing.assert_array_equal(
            pred_std,
            expected_pred,
        )

        np.testing.assert_array_equal(
            emb,
            expected_emb,
        )

        expected_phys = (
            expected_pred
            * self.y_std[None, :]
            + self.y_mean[None, :]
        )

        np.testing.assert_array_equal(
            pred_phys,
            expected_phys,
        )

    def test_batch_size_does_not_change_results(self):
        outputs_1 = predict_real_with_embeddings(
            model=self.model,
            X_real=self.X,
            device=self.device,
            y_mean=self.y_mean,
            y_std=self.y_std,
            batch_size=1,
        )

        outputs_4 = predict_real_with_embeddings(
            model=self.model,
            X_real=self.X,
            device=self.device,
            y_mean=self.y_mean,
            y_std=self.y_std,
            batch_size=4,
        )

        for a, b in zip(
            outputs_1,
            outputs_4,
        ):
            np.testing.assert_array_equal(
                a,
                b,
            )

    def test_output_shapes(self):
        pred_std, pred_phys, emb = (
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X,
                device=self.device,
                y_mean=self.y_mean,
                y_std=self.y_std,
            )
        )

        self.assertEqual(
            pred_std.shape,
            (7, 3),
        )

        self.assertEqual(
            pred_phys.shape,
            (7, 3),
        )

        self.assertEqual(
            emb.shape,
            (7, 3),
        )

    def test_rejects_non_3d_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "X_real must have shape",
        ):
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X[0],
                device=self.device,
                y_mean=self.y_mean,
                y_std=self.y_std,
            )

    def test_rejects_inconsistent_label_statistics(self):
        with self.assertRaisesRegex(
            ValueError,
            "same shape",
        ):
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X,
                device=self.device,
                y_mean=self.y_mean,
                y_std=self.y_std[:2],
            )

    def test_rejects_nonpositive_label_std(self):
        bad_std = self.y_std.copy()
        bad_std[1] = 0.0

        with self.assertRaisesRegex(
            ValueError,
            "positive",
        ):
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X,
                device=self.device,
                y_mean=self.y_mean,
                y_std=bad_std,
            )

    def test_rejects_invalid_batch_size(self):
        with self.assertRaisesRegex(
            ValueError,
            "positive",
        ):
            predict_real_with_embeddings(
                model=self.model,
                X_real=self.X,
                device=self.device,
                y_mean=self.y_mean,
                y_std=self.y_std,
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
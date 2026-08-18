import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paths import (
    DATA_ROOT_ENV_VAR,
    dataset_processed_dir,
    resolve_data_root,
    resolve_processed_artifact,
    resolve_project_root,
)


class TestProjectRootResolution(
    unittest.TestCase
):
    def test_cli_project_root_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_project_root(
                cli_project_root=tmp,
                config_project_root="/does/not/exist",
            )

            self.assertEqual(
                result,
                Path(tmp).resolve(),
            )

    def test_valid_config_project_root_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_project_root(
                config_project_root=tmp,
            )

            self.assertEqual(
                result,
                Path(tmp).resolve(),
            )

    def test_invalid_config_falls_back_to_repo(self):
        result = resolve_project_root(
            config_project_root=(
                "/definitely/not/a/real/path"
            )
        )

        self.assertTrue(
            (result / "src").is_dir()
        )


class TestDataRootResolution(
    unittest.TestCase
):
    def test_cli_overrides_environment_and_config(self):
        with (
            tempfile.TemporaryDirectory() as cli_tmp,
            tempfile.TemporaryDirectory() as env_tmp,
            patch.dict(
                os.environ,
                {
                    DATA_ROOT_ENV_VAR:
                        env_tmp
                },
            ),
        ):
            result = resolve_data_root(
                cli_data_root=cli_tmp,
                config_data_root="/config/path",
            )

            self.assertEqual(
                result,
                Path(cli_tmp).resolve(),
            )

    def test_environment_overrides_config(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    DATA_ROOT_ENV_VAR:
                        tmp
                },
            ),
        ):
            result = resolve_data_root(
                config_data_root="/config/path"
            )

            self.assertEqual(
                result,
                Path(tmp).resolve(),
            )

    def test_config_is_fallback(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            result = resolve_data_root(
                config_data_root="/some/data"
            )

        self.assertEqual(
            result,
            Path("/some/data").resolve(),
        )

    def test_missing_data_root_raises(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            with self.assertRaises(
                ValueError
            ):
                resolve_data_root()


class TestProcessedArtifactResolution(
    unittest.TestCase
):
    def test_dataset_processed_dir(self):
        result = dataset_processed_dir(
            "/data/root",
            "dataset_a",
        )

        self.assertEqual(
            result,
            Path(
                "/data/root/processed/"
                "dataset_a"
            ),
        )

    def test_prefers_dataset_specific_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            canonical = (
                root
                / "processed"
                / "dataset_a"
                / "artifact.npz"
            )

            canonical.parent.mkdir(
                parents=True
            )

            canonical.touch()

            legacy = (
                root
                / "processed"
                / "artifact.npz"
            )

            legacy.touch()

            result = (
                resolve_processed_artifact(
                    data_root=root,
                    dataset_id="dataset_a",
                    file_name="artifact.npz",
                )
            )

            self.assertEqual(
                result,
                canonical,
            )

    def test_legacy_flat_layout_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            legacy = (
                root
                / "processed"
                / "artifact.npz"
            )

            legacy.parent.mkdir(
                parents=True
            )

            legacy.touch()

            result = (
                resolve_processed_artifact(
                    data_root=root,
                    dataset_id="dataset_a",
                    file_name="artifact.npz",
                )
            )

            self.assertEqual(
                result,
                legacy,
            )


if __name__ == "__main__":
    unittest.main()
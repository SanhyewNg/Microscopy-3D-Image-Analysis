import argparse
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from marshmallow.exceptions import ValidationError

from clb.virtules.classify_cd8 import main as classify_cd8_
from clb.virtules.classify_ki67 import main as classify_ki67_
from clb.virtules.classify_panCK import main as classify_panCK_
from clb.virtules.classify_pdl1 import main as classify_pdl1_
from clb.virtules.enhancement import ENHANCEMENT_MAPPING, get_image_shape
from clb.virtules.enhancement import main as enhancement_
from clb.virtules.export_imaris import main as export_imaris_
from clb.virtules.segment import main as segment_
from clb.virtules.spatial_stats import main as spatial_stats_
from clb.virtules.stats_nuclei import main as stats_nuclei_
from clb.virtules.utils import MissingEnvException

ENVS = {"AZ_BATCH_JOB_PREP_WORKING_DIR": "AZ_DIR"}


def get_env(env_name):
    return ENVS[env_name]


class TestExportImaris(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.export_imaris.export")
    def test_minimal(self, export, env):
        env.__getitem__.side_effect = get_env
        export_imaris_(str(self.test_data / "export_imaris.json"))
        export.assert_called_with(
            inputs=[{"path": "AZ_DIR/data/source_series/series0", "name": "DAPI", "channel": 0, "color": "#0000ff"}],
            output_path="AZ_DIR/data/results/random_dir/output.ims",
        )

    @patch("os.environ")
    @patch("clb.virtules.export_imaris.export")
    def test_minimal(self, export, env):
        env.__getitem__.side_effect = get_env
        export_imaris_(str(self.test_data / "export_imaris_real.json"))
        export.assert_called_with(
            inputs=[
                {"path": "AZ_DIR/data/source_series/series0", "name": "DAPI", "channel": 0, "color": "#0000ff"},
                {"path": "AZ_DIR/data/source_series/series0", "name": "CD3_/_CD8", "channel": 3, "color": "#ff0000"},
                {
                    "path": "AZ_DIR/data/results/5d776418e44nl2tu8ls38tzmbpdvpth5/labels.tif",
                    "name": "Segmentation_1_(DAPI)_(job_id:_5d776418e44nl2tu8ls38tzmbpdvpth5)",
                    "channel": 0,
                    "color": None,
                },
                {
                    "path": "AZ_DIR/data/results/5d806e02s1nzna6w8vch6b32pr7dtm7e/labels.tif",
                    "name": "Segmentation_7_(CD3)_(job_id:_5d806e02s1nzna6w8vch6b32pr7dtm7e)",
                    "channel": 0,
                    "color": None,
                },
                {
                    "path": "AZ_DIR/data/results/5d776b78bqcgq4stxmmidp2oshvu4cs8/classification.tif",
                    "name": "Classification_1_(PanCK)_(job_id:_5d776b78bqcgq4stxmmidp2oshvu4cs8)",
                    "channel": 0,
                    "color": "#0000ff",
                },
                {
                    "path": "AZ_DIR/data/results/5d806ddc2gqx65plkao77zzbyw3p3kf3/classification.tif",
                    "name": "Classification_8_(CD8)_(job_id:_5d806ddc2gqx65plkao77zzbyw3p3kf3)",
                    "channel": 0,
                    "color": "#00ff00",
                },
            ],
            output_path="AZ_DIR/data/results/5d80899f20wfgvwdah2qkw1all6hboy9/output.ims",
        )


class TestStatsNuclei(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.stats_nuclei.stats")
    @patch("clb.virtules.stats_nuclei.plot_scatterplots")
    def test_minimal(self, plot, stats, env):
        env.__getitem__.side_effect = get_env
        stats_nuclei_(str(self.test_data / "stats_nuclei.json"))
        stats.assert_called_with(
            Ki67="results/classification.tif",
            channel_names=["DAPI"],
            channels=[0],
            input="AZ_DIR/data/source_series/series0",
            labels="AZ_DIR/data/labels.tif",
            output="AZ_DIR/data/{name}_",
        )
        plot.assert_called_with(
            channel_names=["DAPI"],
            input="AZ_DIR/data/source_series/series0",
            output_dir="AZ_DIR/data/scatterplots",
            stats_path="AZ_DIR/data/{name}_nuclei_stats.csv",
        )

    @patch("os.environ")
    @patch("clb.virtules.stats_nuclei.stats")
    @patch("clb.virtules.stats_nuclei.plot_scatterplots")
    def test_real(self, plot, stats, env):
        env.__getitem__.side_effect = get_env
        stats_nuclei_(str(self.test_data / "stats_nuclei_real.json"))
        stats.assert_called_with(
            CD8="results/5d806ddc2gqx65plkao77zzbyw3p3kf3/classification.tif",
            Ki67="results/5d776b78bqcgq4stxmmidp2oshvu4cs8/classification.tif",
            channel_names=["pan-CK", "CD3 / CD8"],
            channels=[2, 3],
            input="AZ_DIR/data/source_series/series0",
            labels="AZ_DIR/data/results/5d776418e44nl2tu8ls38tzmbpdvpth5/labels.tif",
            output="AZ_DIR/data/results/5d808702ov11emeod66crqwpo4n1wdqe/{name}_",
        )
        plot.assert_called_with(
            channel_names=["pan-CK", "CD3 / CD8"],
            input="AZ_DIR/data/source_series/series0",
            output_dir="AZ_DIR/data/results/5d808702ov11emeod66crqwpo4n1wdqe/scatterplots",
            stats_path="AZ_DIR/data/results/5d808702ov11emeod66crqwpo4n1wdqe/{name}_nuclei_stats.csv",
        )


class TestSpatialStats(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.spatial_stats.stats")
    def test_minimal(self, stats, env):
        env.__getitem__.side_effect = get_env
        spatial_stats_(str(self.test_data / "spatial_stats.json"))
        stats.assert_called_with(
            filter_double_positives=True,
            input="AZ_DIR/data/source_series/series0",
            labels="AZ_DIR/data/results/random_dir/labels.tif",
            output_data_dir="AZ_DIR/data/results/random_dir/data",
            output_graph="AZ_DIR/data/results/random_dir/graph.png",
            ref_class_path="AZ_DIR/data/results/random_dir/classification.tif",
            ref_plot_name="Empty space distance to Classification 1 (PanCK) cells",
            tested_classes_names=["Distance from Classification 1 (PanCK) to Classification 2 (PanCK)"],
            tested_classes_paths=["AZ_DIR/data/results/random_dir/classification.tif"],
        )

    @patch("os.environ")
    @patch("clb.virtules.spatial_stats.stats")
    def test_real(self, stats, env):
        env.__getitem__.side_effect = get_env
        spatial_stats_(str(self.test_data / "spatial_stats_real.json"))
        stats.assert_called_with(
            filter_double_positives=None,
            input="AZ_DIR/data/source_series/series0",
            labels="AZ_DIR/data/results/5d776418e44nl2tu8ls38tzmbpdvpth5/labels.tif",
            output_data_dir="AZ_DIR/data/results/5d808b833inbkdurrexsekdfes0m7y7s/data",
            output_graph="AZ_DIR/data/results/5d808b833inbkdurrexsekdfes0m7y7s/graph.png",
            ref_class_path="AZ_DIR/data/results/5d776b78bqcgq4stxmmidp2oshvu4cs8/classification.tif",
            ref_plot_name="Empty space distance to Classification 1 (PanCK) cells",
            tested_classes_names=["Distance from Classification 1 (PanCK) to Classification 8 (CD8)"],
            tested_classes_paths=["AZ_DIR/data/results/5d806ddc2gqx65plkao77zzbyw3p3kf3/classification.tif"],
        )


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.classify_cd8.classify_")
    def test_minimal_cd8(self, classify, env):
        env.__getitem__.side_effect = get_env
        classify_cd8_(str(self.test_data / "classification.json"))
        classify.assert_called_with(
            Namespace(
                channel_color="red",
                channel_name="cd8",
                channels="0",
                discrete="binary",
                features_type=None,
                input="AZ_DIR/data/source_series/series0",
                instance_model=None,
                labels="AZ_DIR/data/results/random_dir/labels.tif",
                model="models/classification/model_7_eager_torvalds_2_perc_good_class_cd8.pkl",
                outputs=[
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/classification.tif",
                ],
                series=None,
                start=0,
                stop=None,
                uff_colors=["gray", "red"],
                use_cubes=None,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.classify_cd8.classify_")
    def test_real_cd8(self, classify, env):
        env.__getitem__.side_effect = get_env
        classify_cd8_(str(self.test_data / "classification_real_cd8.json"))
        classify.assert_called_with(
            Namespace(
                channel_color="red",
                channel_name="cd8",
                channels="0",
                discrete="binary",
                features_type=None,
                input="AZ_DIR/data/source_series/series0",
                instance_model=None,
                labels="AZ_DIR/data/results/random_dir/labels.tif",
                model="models/classification/model_7_eager_torvalds_2_perc_good_class_cd8.pkl",
                outputs=[
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/classification.tif",
                ],
                series=None,
                start=0,
                stop=None,
                uff_colors=["gray", "red"],
                use_cubes=None,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.classify_ki67.classify_")
    def test_minimal_ki67(self, classify, env):
        env.__getitem__.side_effect = get_env
        classify_ki67_(str(self.test_data / "classification_real_ki67.json"))
        classify.assert_called_with(
            Namespace(
                channel_color="yellow",
                channel_name="Ki67",
                channels="0",
                discrete="binary",
                features_type=None,
                input="AZ_DIR/data/source_series/series0",
                instance_model=None,
                labels="AZ_DIR/data/results/random_dir/labels.tif",
                model="models/classification/model_7_eager_torvalds_3_class_Ki67.pkl",
                outputs=[
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/classification.tif",
                ],
                series=None,
                start=0,
                stop=None,
                uff_colors=["gray", "yellow"],
                use_cubes=None,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.classify_panCK.classify_")
    def test_minimal_panCK(self, classify, env):
        env.__getitem__.side_effect = get_env
        classify_panCK_(str(self.test_data / "classification_real_panCK.json"))
        classify.assert_called_with(
            Namespace(
                channel_color="green",
                channel_name="Epith",
                channels="2",
                discrete="binary",
                features_type=None,
                input="AZ_DIR/data/source_series/series0",
                instance_model=None,
                labels="AZ_DIR/data/results/random_dir/labels.tif",
                model="models/classification/model_7_eager_torvalds_3_class_epith.pkl",
                outputs=[
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/classification.tif",
                ],
                series=None,
                start=0,
                stop=None,
                uff_colors=["gray", "lime"],
                use_cubes=None,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.classify_pdl1.classify_")
    def test_minimal_pdl1(self, classify, env):
        env.__getitem__.side_effect = get_env
        classify_pdl1_(str(self.test_data / "classification_real_pdl1.json"))
        classify.assert_called_with(
            Namespace(
                channel_color="yellow",
                channel_name="pdl1",
                channels="0,2",
                discrete="binary",
                features_type=None,
                input="AZ_DIR/data/source_series/series0",
                instance_model=None,
                labels="AZ_DIR/data/results/random_dir/labels.tif",
                model="models/classification/model_8_angry_ptolemy_1_preproc_class_pdl1.pkl",
                outputs=[
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/classification.tif",
                ],
                series=None,
                start=0,
                stop=None,
                uff_colors=["gray", "yellow"],
                use_cubes=None,
            )
        )

    def test_validation_fail_additional_args_cd8(self):
        with self.assertRaises(ValidationError):
            classify_cd8_(str(self.test_data / "classification_unknown.json"))

    def test_validation_fail_additional_args_ki67(self):
        with self.assertRaises(ValidationError):
            classify_ki67_(str(self.test_data / "classification_unknown.json"))

    def test_validation_fail_additional_args_panCK(self):
        with self.assertRaises(ValidationError):
            classify_panCK_(str(self.test_data / "classification_unknown.json"))

    def test_validation_fail_additional_args_pdl1(self):
        with self.assertRaises(ValidationError):
            classify_pdl1_(str(self.test_data / "classification_unknown.json"))


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.segment.segment")
    def test_minimal(self, segment, env):
        env.__getitem__.side_effect = get_env
        segment_(str(self.test_data / "segment.json"))
        segment.assert_called_with(
            Namespace(
                desired_pixel_size=(0.5, 0.5),
                input="AZ_DIR/data/source_series/series0",
                model="models/model_8_angry_ptolemy.h5",
                no_pixel_resize=False,
                outputs=[
                    "AZ_DIR/data/results/random_dir/labels.tif",
                    "AZ_DIR/data/results/random_dir/series0",
                    "AZ_DIR/data/results/random_dir/{name}.ims",
                ],
                pixel_size=(0.5, 0.5),
                resize_tolerance=0.05,
                series=None,
                start=0,
                stop=None,
                use_channel=0,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.segment.segment")
    def test_segment(self, segment, env):
        env.__getitem__.side_effect = get_env
        segment_(str(self.test_data / "segment_additional.json"))
        segment.assert_called_with(
            Namespace(
                desired_pixel_size=(0.5, 0.5),
                input="AZ_DIR/input_path",
                model="models/model_8_angry_ptolemy.h5",
                no_pixel_resize=False,
                outputs=[
                    "AZ_DIR/output_path/labels.tif",
                    "AZ_DIR/output_path/series0",
                    "AZ_DIR/output_path/{name}.ims",
                ],
                pixel_size=(0.5, 0.5),
                resize_tolerance=0.05,
                series=None,
                start=2,
                stop=10,
                use_channel=0,
            )
        )

    @patch("os.environ")
    @patch("clb.virtules.segment.segment")
    def test_segment_real(self, segment, env):
        env.__getitem__.side_effect = get_env
        segment_(str(self.test_data / "segment_real.json"))
        segment.assert_called_with(
            Namespace(
                desired_pixel_size=(0.5, 0.5),
                input="AZ_DIR/data/source_series/series0",
                model="models/model_8_angry_ptolemy.h5",
                no_pixel_resize=False,
                outputs=[
                    "AZ_DIR/data/results/5d808e65zmb35ejb3jssipjv41oh4byo/labels.tif",
                    "AZ_DIR/data/results/5d808e65zmb35ejb3jssipjv41oh4byo/series0",
                    "AZ_DIR/data/results/5d808e65zmb35ejb3jssipjv41oh4byo/{name}.ims",
                ],
                pixel_size=(0.5, 0.5),
                resize_tolerance=0.05,
                series=None,
                start=0,
                stop=None,
                use_channel=0,
            )
        )

    def test_validation_fail_additional_args(self):
        with self.assertRaises(ValidationError):
            segment_(str(self.test_data / "segment_unknown.json"))


class TestEnhancement(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "virtules"

    @patch("os.environ")
    @patch("clb.virtules.enhancement.denoise")
    @patch("clb.virtules.enhancement.volread")
    def test_minimal(self, volread, denoise, env):
        env.__getitem__.side_effect = get_env
        volread.return_value = np.zeros((300, 300, 3))
        with patch.dict(ENHANCEMENT_MAPPING, {"denoise": denoise}):
            enhancement_(str(self.test_data / "enhancement_denoise.json"))
            denoise.assert_called_once_with(
                channel=0,
                input="AZ_DIR/input_file",
                model="models/denoising/model0.h5",
                output="AZ_DIR/output_file",
                patches_shape=(256, 256),
                patches_stride=(44, 44),
            )

    @patch("clb.virtules.enhancement.volread")
    def test_get_image_shape(self, volread):
        volread.return_value = np.zeros((300, 300, 3))
        self.assertTupleEqual(get_image_shape("dummy_path"), (300, 300))

        volread.return_value = np.zeros((15, 300, 300, 3))
        self.assertTupleEqual(get_image_shape("dummy_path"), (300, 300))


if __name__ == "__main__":
    unittest.main()

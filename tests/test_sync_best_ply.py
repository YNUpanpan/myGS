import importlib.util
import sys
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "scripts" / "sync_best_ply.py"
sync_best_ply = None
if MODULE.exists():
    SPEC = importlib.util.spec_from_file_location("sync_best_ply", MODULE)
    sync_best_ply = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = sync_best_ply
    SPEC.loader.exec_module(sync_best_ply)


class SyncBestPlyTests(unittest.TestCase):
    def test_sync_best_ply_module_exists(self):
        self.assertTrue(MODULE.exists(), f"missing production script: {MODULE}")

    @unittest.skipIf(sync_best_ply is None, "sync script is not implemented")
    def test_best_ply_path_uses_summary_best_iteration(self):
        summary = {
            "best_iteration": 15000,
            "best_metrics": {
                "psnr": 29.488691729168558,
                "ssim": 0.9286100656487221,
                "lpips": 0.11114280185727186,
            },
        }

        artifact = sync_best_ply.build_artifact(
            scene="visible",
            run_id="20260710-082237",
            summary=summary,
            remote_root="/home/pch/myGS",
            local_root=Path("F:/PCH/myGSproj/supersplat_ply"),
        )

        self.assertEqual(
            artifact.remote_ply_path,
            "/home/pch/myGS/outputs/visible/20260710-082237/point_cloud/iteration_15000/point_cloud.ply",
        )
        self.assertEqual(
            artifact.local_path,
            Path(
                "F:/PCH/myGSproj/supersplat_ply/"
                "visible_20260710-082237_best_iter15000_psnr29.4887_ssim0.9286_lpips0.1111.ply"
            ),
        )

    @unittest.skipIf(sync_best_ply is None, "sync script is not implemented")
    def test_missing_best_iteration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "best_iteration"):
            sync_best_ply.build_artifact(
                scene="visible",
                run_id="run",
                summary={"best_metrics": {}},
                remote_root="/home/pch/myGS",
                local_root=Path("supersplat_ply"),
            )

    @unittest.skipIf(sync_best_ply is None, "sync script is not implemented")
    def test_scene_and_run_id_are_sanitized_for_filename(self):
        summary = {
            "best_iteration": 5000,
            "best_metrics": {"psnr": 1.2, "ssim": 0.3, "lpips": 0.4},
        }

        artifact = sync_best_ply.build_artifact(
            scene="visible/test",
            run_id="run:bad",
            summary=summary,
            remote_root="/home/pch/myGS",
            local_root=Path("supersplat_ply"),
        )

        self.assertEqual(
            artifact.local_path.name,
            "visible_test_run_bad_best_iter5000_psnr1.2000_ssim0.3000_lpips0.4000.ply",
        )


if __name__ == "__main__":
    unittest.main()

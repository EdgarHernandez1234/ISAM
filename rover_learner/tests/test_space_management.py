
"""
unit test for space_management.py
tests/test_space_management.py
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import space_management as sm


class TestSpaceManagementBasics(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(sm.human_bytes(0), "0.00 B")
        self.assertEqual(sm.human_bytes(1023), "1023.00 B")
        self.assertEqual(sm.human_bytes(1024), "1.00 KB")
        self.assertEqual(sm.human_bytes(1024 * 1024), "1.00 MB")

    def test_parse_pids_from_text(self):
        txt = "foo 123 bar 9999\npid: 42\n"
        pids = sm.parse_pids_from_text(txt)
        self.assertIn(123, pids)
        self.assertIn(9999, pids)
        self.assertIn(42, pids)

    def test_matches_any_pattern(self):
        cmd = "python3 alpha_trial.py --gst nvarguscamerasrc"
        self.assertTrue(sm.matches_any_pattern(cmd, [r"nvarguscamerasrc"]))
        self.assertFalse(sm.matches_any_pattern(cmd, [r"\bgst-launch-1\.0\b"]))

    def test_is_dangerous_base_dir(self):
        self.assertTrue(sm.is_dangerous_base_dir(Path("/")))
        self.assertTrue(sm.is_dangerous_base_dir(Path.home()))
        # /tmp is generally safe by our guard rails
        self.assertFalse(sm.is_dangerous_base_dir(Path("/tmp/demo_artifacts")))


class TestRunFolderDetection(unittest.TestCase):
    def test_collect_run_folders_flat(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run1 = base / "run_001"
            run2 = base / "run_002"
            other = base / "notes"

            run1.mkdir()
            run2.mkdir()
            other.mkdir()

            (run1 / "log.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (run2 / "metadata.json").write_text("{}", encoding="utf-8")
            (other / "readme.md").write_text("hello", encoding="utf-8")

            mode, collected = sm.collect_run_folders(base, match=None)
            self.assertEqual(mode, "flat")
            paths = sorted([rf.path.name for rf in collected])  # type: ignore[arg-type]
            self.assertEqual(paths, ["run_001", "run_002"])

    def test_collect_run_folders_grouped(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            alpha = base / "alpha_trial"
            beta = base / "beta_trial"
            alpha.mkdir()
            beta.mkdir()

            a1 = alpha / "run_001"
            a2 = alpha / "run_002"
            b1 = beta / "run_010"
            a1.mkdir()
            a2.mkdir()
            b1.mkdir()

            (a1 / "out.csv").write_text("x\n", encoding="utf-8")
            (a2 / "out.csv").write_text("x\n", encoding="utf-8")
            (b1 / "out.csv").write_text("x\n", encoding="utf-8")

            mode, grouped = sm.collect_run_folders(base, match=None)
            self.assertEqual(mode, "grouped")
            self.assertTrue(len(grouped) >= 2)  # type: ignore[arg-type]

            group_names = sorted([g[0].name for g in grouped])  # type: ignore[index]
            self.assertEqual(group_names, ["alpha_trial", "beta_trial"])


class TestPruningLogic(unittest.TestCase):
    def test_prune_run_list_keeps_newest(self):
        runs = [
            sm.RunFolder(Path("/tmp/run_old"), last_ts=10.0, size_bytes=100),
            sm.RunFolder(Path("/tmp/run_mid"), last_ts=20.0, size_bytes=200),
            sm.RunFolder(Path("/tmp/run_new"), last_ts=30.0, size_bytes=300),
        ]

        deleted = []

        def fake_delete(p: Path, dry_run: bool) -> int:
            deleted.append(p.name)
            return 999

        with patch.object(sm, "delete_path", side_effect=fake_delete):
            freed = sm.prune_run_list(runs, keep_last=2, dry_run=False)

        # Should delete only the oldest (run_old)
        self.assertEqual(deleted, ["run_old"])
        self.assertEqual(freed, 999)

    def test_prune_run_list_dry_run(self):
        runs = [
            sm.RunFolder(Path("/tmp/run1"), last_ts=1.0, size_bytes=10),
            sm.RunFolder(Path("/tmp/run2"), last_ts=2.0, size_bytes=10),
            sm.RunFolder(Path("/tmp/run3"), last_ts=3.0, size_bytes=10),
        ]

        with patch.object(sm, "delete_path", return_value=123) as dp:
            freed = sm.prune_run_list(runs, keep_last=1, dry_run=True)

        # Even in dry-run, prune_run_list calls delete_path with dry_run=True
        self.assertEqual(dp.call_count, 2)
        # freed is sum of mocked return values
        self.assertEqual(freed, 246)


class TestCameraCleanup(unittest.TestCase):
    @patch.object(sm, "kill_pids")
    @patch.object(sm, "list_process_table")
    @patch.object(sm, "pids_using_device")
    def test_cleanup_camera_conservative(self, pids_using_device, list_process_table, kill_pids):
        # Two PIDs hold /dev/video0
        pids_using_device.return_value = {100, 200}
        # Only PID 100 looks like gstreamer
        list_process_table.return_value = {
            100: "gst-launch-1.0 nvarguscamerasrc ! fakesink",
            200: "python3 unrelated_camera_thing.py",
            300: "nvarguscamerasrc (but not using /dev/video0 in this test)",
        }

        sm.cleanup_camera_gstreamer(
            camera_devices=["/dev/video0"],
            media_patterns=[r"\bgst-launch-1\.0\b", r"\bnvarguscamerasrc\b"],
            dry_run=True,
            force_kill_camera_users=False,
            restart_nvargus_daemon=False,
        )

        # Conservative mode should target only intersection: PID 100
        kill_pids.assert_called_once()
        args, kwargs = kill_pids.call_args
        targeted = set(args[0])
        self.assertEqual(targeted, {100})

    @patch.object(sm, "kill_pids")
    @patch.object(sm, "list_process_table")
    @patch.object(sm, "pids_using_device")
    def test_cleanup_camera_force_kill(self, pids_using_device, list_process_table, kill_pids):
        pids_using_device.return_value = {100, 200}
        list_process_table.return_value = {
            100: "gst-launch-1.0 nvarguscamerasrc ! fakesink",
            200: "python3 unrelated_camera_thing.py",
        }

        sm.cleanup_camera_gstreamer(
            camera_devices=["/dev/video0"],
            media_patterns=[r"\bgst-launch-1\.0\b", r"\bnvarguscamerasrc\b"],
            dry_run=True,
            force_kill_camera_users=True,
            restart_nvargus_daemon=False,
        )

        kill_pids.assert_called_once()
        args, kwargs = kill_pids.call_args
        targeted = set(args[0])
        self.assertEqual(targeted, {100, 200})

    @patch.object(sm, "which", return_value=None)
    def test_restart_nvargus_daemon_without_systemctl(self, which_mock):
        # Should not crash if systemctl is missing
        sm.cleanup_camera_gstreamer(
            camera_devices=["/dev/video0"],
            media_patterns=[r"nvarguscamerasrc"],
            dry_run=False,
            force_kill_camera_users=False,
            restart_nvargus_daemon=True,
        )


if __name__ == "__main__":
    unittest.main()
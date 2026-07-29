"""Regression tests for the remediated critical audit findings."""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import setup_chinese_tts
import speed_dial


class SpeedDialPersistenceTests(unittest.TestCase):
    def test_mutations_are_concurrent_and_preserve_corrupt_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_path = Path(temp_dir) / "presets.json"
            with mock.patch.object(speed_dial, "SPEED_DIAL_FILE", preset_path):
                threads = [
                    threading.Thread(
                        target=speed_dial.save_preset,
                        args=(f"preset_{index}", "af_bella", f"text {index}")
                    )
                    for index in range(20)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(len(speed_dial.load_presets()), 20)
                json.loads(preset_path.read_text(encoding="utf-8"))

                preset_path.write_text('{"bad": []}', encoding="utf-8")
                original = preset_path.read_bytes()
                self.assertFalse(
                    speed_dial.save_preset("new", "af_bella", "do not overwrite")
                )
                self.assertEqual(preset_path.read_bytes(), original)

                preset_path.write_text(
                    '{"existing":{"voice":"af_bella","text":"safe"}}',
                    encoding="utf-8"
                )
                original = preset_path.read_bytes()
                with mock.patch.object(
                    speed_dial.os, "replace", side_effect=OSError("injected")
                ):
                    self.assertFalse(
                        speed_dial.save_preset("new", "af_bella", "still safe")
                    )
                self.assertEqual(preset_path.read_bytes(), original)
                self.assertEqual(list(Path(temp_dir).glob(".speed_dial.*.tmp")), [])


class ChineseSetupTests(unittest.TestCase):
    def test_downloads_matching_config_and_flat_voice_atomically(self):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            downloaded = Path(kwargs["local_dir"]) / kwargs["filename"]
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"artifact")
            return str(downloaded)

        hub = types.SimpleNamespace(hf_hub_download=fake_download)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            sys.modules, {"huggingface_hub": hub}
        ), mock.patch.object(setup_chinese_tts, "CHINESE_VOICES", ["zf_test.pt"]):
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                with mock.patch.object(
                    setup_chinese_tts, "VOICES_DIR", Path(temp_dir) / "voices"
                ):
                    self.assertTrue(setup_chinese_tts.download_config())
                    Path("voices").mkdir()
                    Path("voices/zf_test.pt").touch()
                    self.assertEqual(setup_chinese_tts.download_voices(), (1, 0))
                    self.assertTrue(Path("config-v1_1-zh.json").is_file())
                    self.assertTrue(Path("voices/zf_test.pt").is_file())
                    self.assertFalse(Path("voices/voices/zf_test.pt").exists())
            finally:
                os.chdir(old_cwd)

        self.assertEqual(calls[0]["repo_id"], "hexgrad/Kokoro-82M-v1.1-zh")


class ModelConstructionTests(unittest.TestCase):
    def test_explicit_checkpoint_and_matching_config_reach_pipeline(self):
        class FakeTensor:
            def to(self, device):
                return self

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.load = lambda *args, **kwargs: FakeTensor()
        fake_torch.from_numpy = lambda value: FakeTensor()
        fake_torch.cat = lambda values, dim=0: FakeTensor()

        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = type("ndarray", (), {})
        constructed = []

        class FakeKModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                constructed.append(self)

            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeKPipeline:
            def __init__(self, lang_code, repo_id=None, model=True, **kwargs):
                self.lang_code = lang_code
                self.repo_id = repo_id
                self.model = model
                self.voices = {}

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KModel = FakeKModel
        fake_kokoro.KPipeline = FakeKPipeline

        spec = importlib.util.spec_from_file_location(
            "models_critical_test", Path(__file__).with_name("models.py")
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "numpy": fake_numpy, "kokoro": fake_kokoro}
        ):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"KOKORO_BASE_DIR": temp_dir}, clear=False
        ):
            root = Path(temp_dir)
            checkpoint = root / "custom-zh.pth"
            checkpoint.write_bytes(b"checkpoint")
            chinese_config = root / "config-v1_1-zh.json"
            chinese_config.write_text("{}", encoding="utf-8")
            voices = root / "voices"
            voices.mkdir()
            (voices / "zf_test.pt").write_bytes(b"voice")
            module.initialize_phonemizer = lambda language: True
            module.download_voice_files = lambda **kwargs: ["zf_test.pt"]

            pipeline = module.build_model(str(checkpoint), "cpu", lang_code="z")
            self.assertIs(pipeline.model, constructed[-1])
            self.assertEqual(constructed[-1].kwargs["model"], str(checkpoint))
            self.assertEqual(
                constructed[-1].kwargs["config"], str(chinese_config)
            )

            override = root / "custom-config.json"
            override.write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {"KOKORO_CONFIG_PATH": str(override)}):
                replacement = module.build_model(str(checkpoint), "cpu", lang_code="z")
            self.assertIsNot(replacement, pipeline)
            self.assertEqual(constructed[-1].kwargs["config"], str(override))

            (root / "kokoro-v1_0.pth").write_bytes(b"english")
            (root / "kokoro-v1_1-zh.pth").write_bytes(b"chinese")
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "af_test.pt").write_bytes(b"unused")
            module.download_voice_files = lambda **kwargs: ["zf_test.pt"]
            english = module.build_model(None, "cpu", lang_code="a")
            chinese = module.build_model(None, "cpu", lang_code="z")
            self.assertIsNot(english, chinese)
            self.assertEqual(
                english.model.kwargs["model"], str(root / "kokoro-v1_0.pth")
            )
            self.assertEqual(
                chinese.model.kwargs["model"], str(root / "kokoro-v1_1-zh.pth")
            )


if __name__ == "__main__":
    unittest.main()

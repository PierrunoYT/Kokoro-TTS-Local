"""Regression tests for the remediated critical audit findings."""

import importlib.util
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from kokoro_tts_local import console, setup_chinese_tts, speed_dial
from kokoro_tts_local.chinese_config import ChineseTextProcessor


class ChineseTextNormalizationTests(unittest.TestCase):
    def test_newlines_survive_normalization(self):
        """Every synthesis path splits on r'\\n+'; collapsing them truncates."""
        normalize = ChineseTextProcessor.normalize_chinese_text

        text = "第一段。\n\n第二段。\n第三段。"
        self.assertEqual(normalize(text).count("\n"), 2)
        self.assertEqual(
            normalize(text).split("\n"),
            ["第一段。", "第二段。", "第三段。"],
        )

        # Intra-line whitespace still collapses, and blank lines do not
        # survive as empty segments.
        self.assertEqual(normalize("你好   世界\n\n\n  再见  "), "你好 世界\n再见")
        # Punctuation spacing still applies, per line.
        self.assertEqual(normalize("你好，世界\n再见"), "你好， 世界\n再见")
        self.assertEqual(normalize("   "), "")


class ConsoleEncodingTests(unittest.TestCase):
    def test_legacy_code_page_is_upgraded_and_utf8_left_alone(self):
        legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with mock.patch.object(sys, "stdout", legacy), mock.patch.object(
            sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        ):
            self.assertTrue(console.enable_utf8_console())
            self.assertEqual(legacy.encoding, "utf-8")
            # The bilingual entry points print this on every line.
            print("下载配置文件 (Downloading Config File)...")

        already_utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with mock.patch.object(sys, "stdout", already_utf8), mock.patch.object(
            sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        ):
            self.assertFalse(console.enable_utf8_console())


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
        ), mock.patch.dict(
            os.environ, {"KOKORO_BASE_DIR": temp_dir}, clear=False
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
            "kokoro_tts_local.models_critical_test", SRC / "kokoro_tts_local" / "models.py"
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
            module.download_voice_files = lambda **kwargs: ["zf_test.pt"]

            pipeline = module.build_model(str(checkpoint), "cpu", lang_code="z")
            self.assertIs(pipeline.model, constructed[-1])
            self.assertEqual(constructed[-1].kwargs["model"], str(checkpoint))
            self.assertEqual(constructed[-1].kwargs["config"], {})

            override = root / "custom-config.json"
            override.write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {"KOKORO_CONFIG_PATH": str(override)}):
                replacement = module.build_model(str(checkpoint), "cpu", lang_code="z")
            self.assertIsNot(replacement, pipeline)
            self.assertEqual(constructed[-1].kwargs["config"], {})

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

            revision = "release/test"
            revision_id = hashlib.sha256(
                f"hexgrad/Kokoro-82M@{revision}".encode("utf-8")
            ).hexdigest()[:16]
            revision_dir = root / "revisions" / revision_id
            revision_dir.mkdir(parents=True)
            revision_model = revision_dir / "kokoro-v1_0.pth"
            revision_config = revision_dir / "config.json"
            revision_model.write_bytes(b"revision checkpoint")
            revision_config.write_text('{"revision": true}', encoding="utf-8")
            versioned = module.build_model(
                None, "cpu", repo_version=revision, lang_code="a"
            )
            self.assertEqual(versioned.model.kwargs["model"], str(revision_model))
            self.assertEqual(versioned.model.kwargs["config"], {"revision": True})

            # A mistyped explicit checkpoint must fail loudly rather than be
            # filled in with the repository default.
            missing = root / "typo.pth"
            with self.assertRaises(ValueError) as caught:
                module.build_model(str(missing), "cpu", lang_code="a")
            self.assertIn("not found", str(caught.exception))
            self.assertFalse(missing.exists())

            # Voice names the router cannot classify never reach a picker.
            (voices / "af_test.pt").write_bytes(b"voice")
            (voices / "custom.pt").write_bytes(b"voice")
            (voices / "qq_test.pt").write_bytes(b"voice")
            listed = module.list_available_voices()
            self.assertIn("af_test", listed)
            self.assertNotIn("custom", listed)
            self.assertNotIn("qq_test", listed)


class ConcurrencyLifecycleTests(unittest.TestCase):
    def test_registry_iteration_validation_and_shutdown(self):
        """Exercise registry identity, family locking, and terminal shutdown."""
        model_calls = []
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        class FakeTensor:
            def to(self, device):
                return self

        # Records whether inference actually ran with gradients disabled.
        grad_disabled = []

        class FakeNoGrad:
            def __enter__(self):
                grad_disabled.append(True)
                return self

            def __exit__(self, *exc):
                grad_disabled.pop()
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.load = mock.Mock(return_value=FakeTensor())
        fake_torch.from_numpy = lambda value: FakeTensor()
        fake_torch.cat = lambda values, dim=0: FakeTensor()
        fake_torch.no_grad = FakeNoGrad
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = type("ndarray", (), {})

        class FakeKModel:
            def __init__(self, **kwargs):
                model_calls.append(self)
                self.kwargs = kwargs

            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeKPipeline:
            def __init__(self, lang_code, repo_id=None, model=True, **kwargs):
                self.lang_code = lang_code
                self.model = model
                self.voices = {}

            def __call__(self, *args, **kwargs):
                nonlocal active, max_active
                assert grad_disabled, "inference ran with autograd enabled"
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    yield ("text", "phonemes", FakeTensor())
                    # Keeping the generator suspended here verifies that the
                    # family lock spans the complete lazy iteration.
                    yield ("text2", "phonemes2", FakeTensor())
                finally:
                    with active_lock:
                        active -= 1

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KModel = FakeKModel
        fake_kokoro.KPipeline = FakeKPipeline
        name = f"kokoro_tts_local.models_concurrency_test_{id(self)}"
        spec = importlib.util.spec_from_file_location(
            name, SRC / "kokoro_tts_local" / "models.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {
            "torch": fake_torch, "numpy": fake_numpy, "kokoro": fake_kokoro
        }):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"KOKORO_BASE_DIR": temp_dir}, clear=False
        ):
            root = Path(temp_dir)
            for filename in ("kokoro-v1_0.pth", "kokoro-v1_1-zh.pth",
                             "config.json", "config-v1_1-zh.json"):
                (root / filename).write_text("{}", encoding="utf-8")
            voices = root / "voices"
            voices.mkdir()
            for filename in ("af_test.pt", "bf_test.pt", "zf_test.pt"):
                (voices / filename).write_bytes(b"voice")
            module.download_voice_files = lambda **kwargs: ["af_test.pt"]

            results = []
            threads = [threading.Thread(
                target=lambda: results.append(module.build_model(None, "cpu", lang_code="a"))
            ) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(results), 12)
            self.assertTrue(all(item is results[0] for item in results))
            self.assertEqual(len(model_calls), 1)

            american = results[0]
            british = module.build_model(None, "cpu", lang_code="b")
            chinese = module.build_model(None, "cpu", lang_code="z")
            self.assertIsNot(american, british)
            self.assertIs(american.model, british.model)
            self.assertIs(american._family_lock, british._family_lock)
            self.assertIsNot(american.model, chinese.model)

            with self.assertRaises(ValueError):
                module.get_language_code_from_voice("xx_test")
            with self.assertRaises(ValueError):
                module.get_language_code_from_voice("aftest")
            with self.assertRaises(ValueError):
                american.iter_speech("hello", voice=str(voices / "bf_test.pt"))

            american.voices["af_test"] = FakeTensor()
            american.load_voice(str(voices / "af_test.pt"))
            fake_torch.load.assert_not_called()

            # A generator abandoned part-way must release the family lock as
            # soon as it is closed, not whenever the caller happens to return.
            partial = american.iter_speech("partial", voice=str(voices / "af_test.pt"))
            next(partial)
            partial.close()
            self.assertTrue(american._family_lock.acquire(timeout=1))
            american._family_lock.release()

            first_started = threading.Event()
            release_first = threading.Event()

            def consume_first():
                generator = american.iter_speech("one", voice=str(voices / "af_test.pt"))
                next(generator)
                first_started.set()
                release_first.wait(2)
                list(generator)

            first = threading.Thread(target=consume_first)
            first.start()
            self.assertTrue(first_started.wait(1))
            second_done = threading.Event()
            second_errors = []

            def consume_second():
                try:
                    list(british.iter_speech("two", voice=str(voices / "bf_test.pt")))
                except RuntimeError as exc:
                    second_errors.append(exc)
                finally:
                    second_done.set()

            second = threading.Thread(target=consume_second)
            second.start()
            self.assertFalse(second_done.wait(.05))

            shutdown_done = []
            shutdown_threads = [threading.Thread(target=lambda: (
                module.shutdown_pipelines(), shutdown_done.append(True)
            )) for _ in range(3)]
            for thread in shutdown_threads:
                thread.start()
            self.assertFalse(any(not thread.is_alive() for thread in shutdown_threads))
            with self.assertRaises(RuntimeError):
                module.build_model(None, "cpu", lang_code="a")
            release_first.set()
            first.join(2)
            second.join(2)
            for thread in shutdown_threads:
                thread.join(2)
            self.assertEqual(len(shutdown_done), 3)
            self.assertEqual(len(second_errors), 1)
            self.assertEqual(max_active, 1)
            self.assertTrue(american._closed)
            self.assertIsNone(american.model)
            with self.assertRaises(RuntimeError):
                next(american.iter_speech("old", voice=str(voices / "af_test.pt")))
            module.shutdown_pipelines()  # terminal and idempotent


if __name__ == "__main__":
    unittest.main()

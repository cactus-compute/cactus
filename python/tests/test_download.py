import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cactus.cli.download import (
    archives_from_repo_files,
    combo_label,
    parse_combo,
    promote_single_root,
    resolve_archive,
    resolve_cq_repo,
    safe_extract_archive,
    suggested_cq_repo,
    validate_extracted_cq,
    verify_archive_sha256,
)


class TestCqDownloadResolver(unittest.TestCase):
    def test_resolve_requires_cactus_compute_cq_repo(self):
        self.assertEqual(
            resolve_cq_repo("Cactus-Compute/lfm2p5-350m-cq"),
            ("Cactus-Compute/lfm2p5-350m-cq", "lfm2p5-350m"),
        )
        with self.assertRaisesRegex(RuntimeError, "try Cactus-Compute/LFM2.5-350M-cq"):
            resolve_cq_repo("LiquidAI/LFM2.5-350M")

    def test_suggested_cq_repo_replaces_org_and_adds_suffix(self):
        self.assertEqual(
            suggested_cq_repo("LiquidAI/LFM2.5-350M"),
            "Cactus-Compute/LFM2.5-350M-cq",
        )
        self.assertEqual(
            suggested_cq_repo("OtherOrg/model-cq"),
            "Cactus-Compute/model-cq",
        )

    def test_parse_combo_is_order_insensitive(self):
        self.assertEqual(parse_combo("L4V3A4.zip"), {"L": 4, "V": 3, "A": 4})
        self.assertEqual(parse_combo("A4V4L1.zip"), {"A": 4, "V": 4, "L": 1})

    def test_parse_combo_rejects_loose_or_duplicate_names(self):
        self.assertEqual(parse_combo("model-L4.zip"), {})
        self.assertEqual(parse_combo("L4L3.zip"), {})
        self.assertEqual(parse_combo("L4-extra.zip"), {})

    def test_strict_combo_match(self):
        archives = archives_from_repo_files(["L1V4.zip", "L2V4.zip", "L3V4.zip", "L4V4.zip", "model-L4V4.zip"])
        resolution = resolve_archive(
            "Cactus-Compute/lfm2p5-vl-1p6b-cq",
            "lfm2.5-vl-1.6b",
            archives,
            {"L": 4, "V": 4},
        )
        self.assertEqual(resolution.archive.filename, "L4V4.zip")
        self.assertEqual(combo_label(resolution.archive.combo), "L4V4")

    def test_missing_supported_combo_errors(self):
        archives = archives_from_repo_files(["L1V4.zip", "L2V4.zip", "L3V4.zip", "L4V4.zip"])
        with self.assertRaisesRegex(RuntimeError, "Available combos: L1V4, L2V4, L3V4, L4V4"):
            resolve_archive(
                "Cactus-Compute/lfm2p5-vl-1p6b-cq",
                "lfm2.5-vl-1.6b",
                archives,
                {"L": 4, "V": 2},
            )

    def test_unsupported_modality_warns_and_ignores(self):
        archives = archives_from_repo_files(["L1.zip", "L2.zip", "L3.zip", "L4.zip"])
        resolution = resolve_archive(
            "Cactus-Compute/lfm2p5-350m-cq",
            "lfm2.5-350m",
            archives,
            {"L": 4, "A": 4},
        )
        self.assertEqual(resolution.archive.filename, "L4.zip")
        self.assertEqual(len(resolution.warnings), 1)
        self.assertIn("has no A modality", resolution.warnings[0])


class TestCqSafeExtraction(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cactus_cq_test_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_minimal_package(self, base: Path):
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.txt").write_text("model_type=test\n", encoding="utf-8")
        (base / "token_embeddings.weights").write_bytes(b"x")
        (base / "vocab.txt").write_text("0\t<pad>\n", encoding="utf-8")
        (base / "tokenizer_config.txt").write_text("tokenizer_type=bpe\nvocab_format=id_tab_token\n", encoding="utf-8")
        (base / "special_tokens.json").write_text("{}", encoding="utf-8")
        (base / "tokenizer.json").write_text("{}", encoding="utf-8")
        (base / "merges.txt").write_text("", encoding="utf-8")

    def test_zip_extract_and_promote_single_root(self):
        package = self.root / "pkg"
        self._write_minimal_package(package / "L4")
        archive = self.root / "L4.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in (package / "L4").iterdir():
                zf.write(path, f"L4/{path.name}")

        out = self.root / "out"
        out.mkdir()
        safe_extract_archive(archive, out)
        promote_single_root(out)
        validate_extracted_cq(out)
        self.assertTrue((out / "config.txt").exists())

    def test_promote_single_root_rejects_ambiguous_layout(self):
        out = self.root / "out"
        (out / "a").mkdir(parents=True)
        (out / "b").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "archive root or under one top-level directory"):
            promote_single_root(out)

    def test_validate_extracted_cq_requires_tokenizer_sidecars(self):
        package = self.root / "pkg"
        self._write_minimal_package(package)
        (package / "tokenizer.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing tokenizer sidecar"):
            validate_extracted_cq(package)

    def test_verify_archive_sha256_rejects_mismatch(self):
        archive = self.root / "L4.zip"
        archive.write_bytes(b"archive")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            verify_archive_sha256(archive, "0" * 64)

    def test_zip_rejects_traversal(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escape.txt", "bad")
        out = self.root / "out"
        out.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
            safe_extract_archive(archive, out)

    def test_tar_rejects_symlink(self):
        archive = self.root / "bad.tar"
        target = self.root / "target.txt"
        target.write_text("x", encoding="utf-8")
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = str(target)
            tf.addfile(info)
        out = self.root / "out"
        out.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Refusing link"):
            safe_extract_archive(archive, out)


if __name__ == "__main__":
    unittest.main()

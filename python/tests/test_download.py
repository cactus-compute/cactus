import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cactus.cli.utils import (
    archives_from_repo_files,
    parse_cq_bits,
    promote_single_root,
    resolve_archive,
    safe_extract_archive,
    suggested_cq_repo,
    validate_extracted_cq,
    verify_archive_sha256,
)


class TestCqDownloadResolver(unittest.TestCase):
    def test_suggested_cq_repo_strips_org(self):
        self.assertEqual(
            suggested_cq_repo("LiquidAI/LFM2.5-350M"),
            "Cactus-Compute/LFM2.5-350M",
        )
        self.assertEqual(
            suggested_cq_repo("google/gemma-4-E2B-it"),
            "Cactus-Compute/gemma-4-E2B-it",
        )

    def test_suggested_cq_repo_strips_existing_cq_suffix(self):
        self.assertEqual(
            suggested_cq_repo("Cactus-Compute/model-cq4"),
            "Cactus-Compute/model",
        )
        self.assertEqual(
            suggested_cq_repo("Cactus-Compute/model-cq"),
            "Cactus-Compute/model",
        )

    def test_parse_cq_bits(self):
        self.assertEqual(parse_cq_bits("gemma-4-E2B-it-cq1.zip"), 1)
        self.assertEqual(parse_cq_bits("gemma-4-E2B-it-cq4.zip"), 4)
        self.assertEqual(parse_cq_bits("gemma-4-E2B-it-CQ3.tar.gz"), 3)
        self.assertIsNone(parse_cq_bits("README.md"))
        self.assertIsNone(parse_cq_bits("L4V4A4.zip"))

    def test_archives_from_repo_files(self):
        files = [
            "gemma-4-E2B-it-cq1.zip",
            "gemma-4-E2B-it-cq2.zip",
            "gemma-4-E2B-it-cq3.zip",
            "gemma-4-E2B-it-cq4.zip",
            "README.md",
            "config.json",
        ]
        sizes = {"gemma-4-E2B-it-cq4.zip": 500_000_000}
        archives = archives_from_repo_files(files, sizes=sizes)
        self.assertEqual(len(archives), 4)
        self.assertEqual(archives[0].bits, 1)
        self.assertEqual(archives[3].bits, 4)
        self.assertEqual(archives[3].size, 500_000_000)
        self.assertIsNone(archives[0].size)

    def test_resolve_archive_finds_match(self):
        files = ["model-cq1.zip", "model-cq2.zip", "model-cq3.zip", "model-cq4.zip"]
        archives = archives_from_repo_files(files)
        resolution = resolve_archive("Cactus-Compute/model", "model", archives, 4)
        self.assertEqual(resolution.archive.filename, "model-cq4.zip")
        self.assertEqual(resolution.archive.bits, 4)

    def test_resolve_archive_missing_bits_errors(self):
        files = ["model-cq1.zip", "model-cq2.zip"]
        archives = archives_from_repo_files(files)
        with self.assertRaisesRegex(RuntimeError, "cq4 not found"):
            resolve_archive("Cactus-Compute/model", "model", archives, 4)

    def test_resolve_archive_empty_errors(self):
        with self.assertRaisesRegex(RuntimeError, "No CQ archives"):
            resolve_archive("Cactus-Compute/model", "model", [], 4)


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
        self._write_minimal_package(package / "model-cq4")
        archive = self.root / "model-cq4.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in (package / "model-cq4").iterdir():
                zf.write(path, f"model-cq4/{path.name}")

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
        archive = self.root / "model-cq4.zip"
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

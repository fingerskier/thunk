"""Tests for the pinned shared tokenizer trainer (script/train_tokenizer.py).

Run:  python script/test_train_tokenizer.py
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_tokenizer as tt  # noqa: E402


CORPUS = [
    "<src:python> <tgt:english> def add(x, y): return x + y <sep> Return x plus y.",
    "<src:english> <tgt:python> Return x plus y. <sep> def add(x, y): return x + y",
    "<src:lean4> <tgt:english> theorem foo : 1 + 1 = 2 := by norm_num <sep> one plus one is two",
    "<src:java> <tgt:csharp> public static int f(int n) { return n; } <sep> public static int f(int n) { return n; }",
    "<src:digits> <tgt:english> 3247 <sep> three thousand two hundred forty seven",
    "<src:bash> <tgt:powershell> rm report.txt <sep> Remove-Item report.txt",
    "    indented\tcode   with   spaces",
] * 40


class ControlTagTests(unittest.TestCase):
    def test_tags_cover_both_models_and_sep(self):
        tags = tt.control_tags()
        self.assertEqual(len(tags), len(set(tags)), "duplicate tags")
        for t in ("<sep>", "<src:english>", "<tgt:english>", "<src:lean4>",
                  "<tgt:lean4>", "<src:csharp>", "<tgt:cpp>", "<src:digits>",
                  "<tgt:powershell>", "<src:lean>"):
            self.assertIn(t, tags)


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        corpus = os.path.join(cls.tmp.name, "corpus.txt")
        with open(corpus, "w", encoding="utf-8") as fh:
            fh.write("\n".join(CORPUS) + "\n")
        cls.out = os.path.join(cls.tmp.name, "v_test", "tokenizer.model")
        cls.result = tt.train([corpus], cls.out, vocab_size=400)
        import sentencepiece as spm
        cls.sp = spm.SentencePieceProcessor()
        cls.sp.load(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        self.assertTrue(os.path.exists(self.out))
        self.assertTrue(os.path.exists(self.out.replace(".model", ".vocab")))
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(self.out),
                                                    "MANIFEST.json")))

    def test_special_ids_fixed(self):
        self.assertEqual(self.sp.pad_id(), 0)
        self.assertEqual(self.sp.unk_id(), 1)
        self.assertEqual(self.sp.bos_id(), 2)
        self.assertEqual(self.sp.eos_id(), 3)

    def test_control_tags_are_single_tokens(self):
        for tag in tt.control_tags():
            pieces = [p for p in self.sp.encode(tag, out_type=str) if p != "▁"]
            self.assertEqual(pieces, [tag], f"{tag} split into {pieces}")

    def test_tagged_line_roundtrip_preserves_whitespace(self):
        text = "<src:python> <tgt:english> def add(x, y):\n    return x + y <sep> Return x plus y."
        ids = self.sp.encode(text, out_type=int)
        self.assertEqual(self.sp.decode(ids), text)

    def test_byte_fallback_for_unseen_chars(self):
        ids = self.sp.encode("λ→∀ 日本", out_type=int)
        self.assertNotIn(self.sp.unk_id(), ids)

    def test_manifest_records_inputs(self):
        self.assertEqual(self.result["vocab_size"], self.sp.get_piece_size())
        self.assertEqual(len(self.result["inputs"]), 1)
        self.assertIn("sha256", self.result["inputs"][0])


if __name__ == "__main__":
    unittest.main()

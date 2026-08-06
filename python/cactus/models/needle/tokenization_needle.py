"""Slow SentencePiece tokenizer for Needle."""

from __future__ import annotations

import os
import shutil
from typing import Any

import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class NeedleTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        tool_call_token: str = "<tool_call>",
        tools_token: str = "<tools>",
        **kwargs: Any,
    ) -> None:
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.sp = self.sp_model
        self.tool_call_token = tool_call_token
        self.tools_token = tools_token
        additional = list(kwargs.pop("additional_special_tokens", []) or [])
        for token in (tool_call_token, tools_token):
            if token not in additional:
                additional.append(token)
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            additional_special_tokens=additional,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.GetPieceSize())

    @property
    def tool_call_token_id(self) -> int:
        return int(self.sp_model.PieceToId(self.tool_call_token))

    @property
    def tools_token_id(self) -> int:
        return int(self.sp_model.PieceToId(self.tools_token))

    def get_vocab(self) -> dict[str, int]:
        vocab = {self.sp_model.IdToPiece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return list(self.sp_model.EncodeAsPieces(text))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.PieceToId(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.IdToPiece(int(index)))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.sp_model.DecodePieces(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            all_ids = list(token_ids_0)
        else:
            all_ids = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        special = {
            self.pad_token_id,
            self.eos_token_id,
            self.bos_token_id,
            self.unk_token_id,
            self.tool_call_token_id,
            self.tools_token_id,
        }
        return [1 if token_id in special else 0 for token_id in all_ids]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        out_name = "tokenizer.model" if filename_prefix is None else f"{filename_prefix}-tokenizer.model"
        out_path = os.path.join(save_directory, out_name)
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_path):
            shutil.copyfile(self.vocab_file, out_path)
        return (out_path,)

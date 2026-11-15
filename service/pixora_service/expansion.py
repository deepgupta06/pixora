"""Turning a short description into a detailed one, with a small local language model.

The model is the GPT-2 fine-tune that Fooocus uses for the same job. It continues
the user's words, and is only allowed to pick from a list of words that improve a
picture (`positive.txt`, which comes with the model), so it adds detail and
quality terms and never changes what the picture is of.
"""

from __future__ import annotations

from pathlib import Path

import torch

# CLIP reads 77 tokens; the rest of a prompt is dropped.
PROMPT_TOKENS = 75


class PromptExpander:
    def __init__(self, folder: Path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
        self.model = AutoModelForCausalLM.from_pretrained(str(folder)).eval()
        words = {w.strip().lower() for w in (folder / "positive.txt").read_text(encoding="utf-8").splitlines()}
        words.discard("")
        allowed = torch.full((len(self.tokenizer),), float("-inf"))
        for token, index in self.tokenizer.get_vocab().items():
            # "Ġ" marks the start of a word: whole listed words only, never fragments.
            if token.startswith("Ġ") and token[1:].lower() in words:
                allowed[index] = 0.0
        self.comma = self.tokenizer.convert_tokens_to_ids(",")
        allowed[self.comma] = 0.0
        self.allowed = allowed

    def expand(self, prompt: str, seed: int, room: int = PROMPT_TOKENS) -> str:
        """`prompt` with detail added, to about `room` tokens in all."""
        from transformers import LogitsProcessor, LogitsProcessorList

        text = " ".join(prompt.split()).rstrip(",. ") + ","
        inputs = self.tokenizer(text, return_tensors="pt")
        new = room - inputs["input_ids"].shape[1]
        if new < 4:
            return prompt
        allowed, comma = self.allowed, self.comma

        class OnlyListedWords(LogitsProcessor):
            def __call__(self, input_ids, scores):
                bias = allowed.clone()
                bias[input_ids[0]] = float("-inf")  # no word twice
                bias[comma] = 0.0
                return scores + bias

        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=new, do_sample=True, top_k=100,
                logits_processor=LogitsProcessorList([OnlyListedWords()]), pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip().rstrip(",")

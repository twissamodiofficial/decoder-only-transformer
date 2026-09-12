import os
from datasets import load_dataset
import re
from tokenizers import Tokenizer, normalizers, pre_tokenizers, Regex, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import modules.config as config

class TokenizeData:
    def __init__(self):
        self.tokenizer = None
        self.tokenizer_path = config.TOKENIZER_PATH

    def load_data(self, split="train"):
        data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        return data[split]

    def train_wikitext_tokenizer(self, vocab_size=config.VOCAB_SIZE, save_path=None):
        if save_path is None:
            save_path = self.tokenizer_path
        
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        train_split = self.load_data()

        def batch_iterator():
            for item in train_split:
                # Skip empty lines to save processing overhead
                if item["text"].strip():
                    yield item["text"]

        self.tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Replace(Regex(r"[\u201c\u201d]"), '"'),
            normalizers.Replace(Regex(r"[\u2018\u2019]"), "'"),
            normalizers.Replace(Regex(r"[\u2013\u2014]"), "-"),
            normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        ) # converts "Héllò hôw are ü?" to "hello how are u?"
        
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
        ) # splits on whitespace and punctuation
        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        
        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=special_tokens,
            end_of_word_suffix="</w>",
        )
        self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )
        self.tokenizer.post_processor = processor

        self.tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        self.tokenizer.save(save_path)
        print("Tokenizer training complete!")
        return self.tokenizer

    def load_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file '{tokenizer_path}' not found.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        return self.tokenizer

    def encode_tokens(self, text):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded.")
        return self.tokenizer.encode(text).ids

    def clean_decode(self, text):
        return re.sub(r'\s+([.,!?;:\')])', r'\1', text)

    def decode_tokens(self, token_ids):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded.")
        decoded_text = self.tokenizer.decode(token_ids)
        return self.clean_decode(decoded_text)


if __name__ == "__main__":
    tokenizer_instance = TokenizeData()
    tokenizer_instance.train_wikitext_tokenizer()
    tokenizer_instance.load_tokenizer()
    sample_text = "Hello, world! This is a test."
    encoded = tokenizer_instance.encode_tokens(sample_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer_instance.decode_tokens(encoded)
    print(f"Decoded: {decoded}")
import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from nlp.best_model import *
import datasets
import os
from icu import BreakIterator, Locale
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


SRC_LANGUAGE = "en"
TRG_LANGUAGE = "my"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "webapp", "Seq2SeqTransformer-additive.pt")

dataset_name = "archx64/english-burmese-parallel"
dataset = datasets.load_dataset(dataset_name)
train = [(row["en"], row["my"]) for row in dataset["train"]]

token_transform = {}
vocab_transform = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def burmese_tokenizer(sentence):
    bi = BreakIterator.createWordInstance(Locale(TRG_LANGUAGE))
    bi.setText(sentence)
    tokens = []
    start = bi.first()
    for end in bi:
        token = sentence[start:end].strip()  # remove leading/trailing spaces
        if token:  # only add non-empty tokens
            tokens.append(token)
        start = end
    return tokens


def yield_tokens(data, language):
    language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}

    for data_sample in data:
        yield token_transform[language](
            data_sample[language_index[language]]
        )  # either first or second index


token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")
token_transform[TRG_LANGUAGE] = burmese_tokenizer
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]


for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train, ln),
        min_freq=2,  # The minimum frequency needed to include a token in the vocabulary. if not, everything will be treated as UNK
        specials=special_symbols,
        special_first=True,
    )  # indicates whether to insert symbols at the beginning or at the end
# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat(
        (torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def generate_text_transform(token_transform, vocab_transform):
    text_transform = {}
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],  # Tokenization
            vocab_transform[ln],  # Numericalization
            tensor_transform,
        )
    return text_transform


text_transform = generate_text_transform(token_transform, vocab_transform)


input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

enc = Encoder(
    input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device
)

dec = Decoder(
    output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, enc_dropout, device
)

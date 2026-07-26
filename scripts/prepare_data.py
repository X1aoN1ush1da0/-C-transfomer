import json
import re
import struct
from pathlib import Path


PAD = 0
BOS = 1
EOS = 2
UNK = 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def english_tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def chinese_tokens(text):
    return [ch for ch in text if not ch.isspace()]


def build_vocab(sentences):
    vocab = {token: i for i, token in enumerate(SPECIAL)}
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def ids(tokens, vocab):
    return [vocab.get(token, UNK) for token in tokens]


def main():
    root = Path(__file__).parent
    corpus_path = root / "data" / "mini_en_zh.tsv"
    output_path = root / "data" / "mini_en_zh.bin"

    pairs = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        english, chinese = line.split("\t")
        pairs.append((english_tokens(english), chinese_tokens(chinese)))

    src_vocab = build_vocab(source for source, _ in pairs)
    tgt_vocab = build_vocab(target for _, target in pairs)

    with output_path.open("wb") as f:
        f.write(struct.pack("iii", len(src_vocab), len(tgt_vocab), len(pairs)))
        for source, target in pairs:
            src_ids = [BOS] + ids(source, src_vocab) + [EOS]
            tgt_input = [BOS] + ids(target, tgt_vocab)
            tgt_output = ids(target, tgt_vocab) + [EOS]
            f.write(struct.pack("ii", len(src_ids), len(tgt_input)))
            f.write(struct.pack(f"{len(src_ids)}i", *src_ids))
            f.write(struct.pack(f"{len(tgt_input)}i", *tgt_input))
            f.write(struct.pack(f"{len(tgt_output)}i", *tgt_output))

    (root / "data" / "src_vocab.json").write_text(
        json.dumps(src_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "data" / "tgt_vocab.json").write_text(
        json.dumps(tgt_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"samples: {len(pairs)}")
    print(f"src vocab: {len(src_vocab)}")
    print(f"tgt vocab: {len(tgt_vocab)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()

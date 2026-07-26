import json
import random
import re
import struct
from collections import Counter
from pathlib import Path
from opencc import OpenCC


PAD = 0
BOS = 1
EOS = 2
UNK = 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def english_tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def chinese_tokens(text):
    return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]


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
    raw_path = root / "data" / "cmn.txt"
    output_prefix = root / "data" / "tatoeba_medium_simplified"
    pairs = []
    seen = set()
    converter = OpenCC("t2s")

    for line in raw_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        source = english_tokens(parts[0])
        target = chinese_tokens(converter.convert(parts[1]))
        if not 2 <= len(source) <= 10 or not 2 <= len(target) <= 18:
            continue
        key = (tuple(source), tuple(target))
        if key not in seen:
            seen.add(key)
            pairs.append((source, target))

    source_count = Counter(token for source, _ in pairs for token in source)
    target_count = Counter(token for _, target in pairs for token in target)
    source_keep = {token for token, _ in source_count.most_common(2200)}
    target_keep = {token for token, _ in target_count.most_common(1800)}
    pairs = [
        (source, target)
        for source, target in pairs
        if all(token in source_keep for token in source)
        and all(token in target_keep for token in target)
    ]
    random.Random(7).shuffle(pairs)
    pairs = pairs[:28000]
    src_vocab = build_vocab(source for source, _ in pairs)
    tgt_vocab = build_vocab(target for _, target in pairs)

    with output_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as f:
        for source, target in pairs:
            f.write(" ".join(source) + "\t" + "".join(target) + "\n")

    with output_prefix.with_suffix(".bin").open("wb") as f:
        f.write(struct.pack("iii", len(src_vocab), len(tgt_vocab), len(pairs)))
        for source, target in pairs:
            src_ids = [BOS] + ids(source, src_vocab) + [EOS]
            tgt_input = [BOS] + ids(target, tgt_vocab)
            tgt_output = ids(target, tgt_vocab) + [EOS]
            f.write(struct.pack("ii", len(src_ids), len(tgt_input)))
            f.write(struct.pack(f"{len(src_ids)}i", *src_ids))
            f.write(struct.pack(f"{len(tgt_input)}i", *tgt_input))
            f.write(struct.pack(f"{len(tgt_output)}i", *tgt_output))

    output_prefix.with_name(output_prefix.name + "_src_vocab.json").write_text(
        json.dumps(src_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_prefix.with_name(output_prefix.name + "_tgt_vocab.json").write_text(
        json.dumps(tgt_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_prefix.with_name(output_prefix.name + "_meta.json").write_text(
        json.dumps({"source": "Tatoeba via ManyThings", "samples": len(pairs)}, indent=2),
        encoding="utf-8",
    )
    print(f"samples: {len(pairs)}")
    print(f"src vocab: {len(src_vocab)}")
    print(f"tgt vocab: {len(tgt_vocab)}")


if __name__ == "__main__":
    main()

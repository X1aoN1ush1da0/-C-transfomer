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
MAX_PAIRS = 120000
SOURCE_VOCAB_LIMIT = 5000
TARGET_VOCAB_LIMIT = 3000


def english_tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def chinese_tokens(text):
    return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]


def valid_pair(source, target):
    if not 2 <= len(source) <= 12 or not 2 <= len(target) <= 22:
        return False
    ratio = len(target) / len(source)
    return 0.8 <= ratio <= 4.0


def pair_stream(root, converter):
    tatoeba_path = root / "data" / "cmn.txt"
    for line in tatoeba_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        source = english_tokens(parts[0])
        target = chinese_tokens(converter.convert(parts[1]))
        if valid_pair(source, target):
            yield source, target

    source_path = root / "data" / "ted2013_en_zh" / "TED2013.en-zh.en"
    target_path = root / "data" / "ted2013_en_zh" / "TED2013.en-zh.zh"
    with source_path.open("r", encoding="utf-8", errors="ignore") as source_file, target_path.open(
        "r", encoding="utf-8", errors="ignore"
    ) as target_file:
        for source_line, target_line in zip(source_file, target_file):
            source = english_tokens(source_line)
            target = chinese_tokens(converter.convert(target_line))
            if valid_pair(source, target):
                yield source, target


def build_vocab(sentences):
    vocab = {token: index for index, token in enumerate(SPECIAL)}
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def ids(tokens, vocab):
    return [vocab.get(token, UNK) for token in tokens]


def main():
    root = Path(__file__).parent
    output_prefix = root / "data" / "mixed_short_simplified"
    converter = OpenCC("t2s")

    source_count = Counter()
    target_count = Counter()
    candidate_count = 0
    for source, target in pair_stream(root, converter):
        source_count.update(source)
        target_count.update(target)
        candidate_count += 1
    print(f"filtered candidates: {candidate_count}")

    source_keep = {token for token, _ in source_count.most_common(SOURCE_VOCAB_LIMIT)}
    target_keep = {token for token, _ in target_count.most_common(TARGET_VOCAB_LIMIT)}
    random_generator = random.Random(7)
    reservoir = []
    kept_count = 0
    for source, target in pair_stream(root, converter):
        if not all(token in source_keep for token in source):
            continue
        if not all(token in target_keep for token in target):
            continue
        kept_count += 1
        pair = (source, target)
        if len(reservoir) < MAX_PAIRS:
            reservoir.append(pair)
            continue
        replace = random_generator.randrange(kept_count)
        if replace < MAX_PAIRS:
            reservoir[replace] = pair
    print(f"vocabulary-filtered candidates: {kept_count}")

    unique = {}
    for source, target in reservoir:
        unique[(tuple(source), tuple(target))] = (source, target)
    pairs = list(unique.values())
    random_generator.shuffle(pairs)
    src_vocab = build_vocab(source for source, _ in pairs)
    tgt_vocab = build_vocab(target for _, target in pairs)

    with output_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as file:
        for source, target in pairs:
            file.write(" ".join(source) + "\t" + "".join(target) + "\n")

    with output_prefix.with_suffix(".bin").open("wb") as file:
        file.write(struct.pack("iii", len(src_vocab), len(tgt_vocab), len(pairs)))
        for source, target in pairs:
            src_ids = [BOS] + ids(source, src_vocab) + [EOS]
            tgt_input = [BOS] + ids(target, tgt_vocab)
            tgt_output = ids(target, tgt_vocab) + [EOS]
            file.write(struct.pack("ii", len(src_ids), len(tgt_input)))
            file.write(struct.pack(f"{len(src_ids)}i", *src_ids))
            file.write(struct.pack(f"{len(tgt_input)}i", *tgt_input))
            file.write(struct.pack(f"{len(tgt_output)}i", *tgt_output))

    output_prefix.with_name(output_prefix.name + "_src_vocab.json").write_text(
        json.dumps(src_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_prefix.with_name(output_prefix.name + "_tgt_vocab.json").write_text(
        json.dumps(tgt_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_prefix.with_name(output_prefix.name + "_meta.json").write_text(
        json.dumps(
            {
                "sources": ["Tatoeba via ManyThings", "OPUS TED2013 v1.1"],
                "samples": len(pairs),
                "candidate_pairs": candidate_count,
                "vocabulary_filtered_pairs": kept_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"samples: {len(pairs)}")
    print(f"source vocab: {len(src_vocab)}")
    print(f"target vocab: {len(tgt_vocab)}")


if __name__ == "__main__":
    main()

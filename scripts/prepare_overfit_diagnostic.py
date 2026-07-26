import struct
from pathlib import Path


root = Path(__file__).parent
source_path = root / "data" / "tatoeba_medium_simplified.bin"
output_path = root / "data" / "medium_overfit.bin"

with source_path.open("rb") as source:
    src_vocab, tgt_vocab, _ = struct.unpack("iii", source.read(12))
    src_len, tgt_len = struct.unpack("ii", source.read(8))
    src = source.read(src_len * 4)
    tgt_input = source.read(tgt_len * 4)
    tgt_output = source.read(tgt_len * 4)

with output_path.open("wb") as output:
    output.write(struct.pack("iii", src_vocab, tgt_vocab, 10))
    for _ in range(10):
        output.write(struct.pack("ii", src_len, tgt_len))
        output.write(src)
        output.write(tgt_input)
        output.write(tgt_output)

print(output_path)

import ijson
import json
import os
from decimal import Decimal

input_file = "data/i24.json"
num_chunks = 20

total_size = os.path.getsize(input_file)
target_size = total_size / num_chunks

chunk_idx = 1
current_size = 0

print(f"Total size: {total_size / (1024**3):.2f} GB")
print(f"Target size per chunk: {target_size / (1024**3):.2f} GB")


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


out_f = open(f"i24_chunk_{chunk_idx:02d}.json", "w")
out_f.write("[\n")
first_item = True

# Open the large file and stream items
with open(input_file, "rb") as f:
    # 'item' prefix catches the elements inside the root array
    items = ijson.items(f, "item")

    for item in items:
        item_str = json.dumps(item, default=decimal_default)
        item_size = len(item_str.encode("utf-8"))

        # If adding this item pushes us over the target size, rotate to next chunk
        # We also make sure we don't go beyond our num_chunks (put all remaining in the last chunk)
        if current_size + item_size > target_size and chunk_idx < num_chunks:
            out_f.write("\n]\n")
            out_f.close()

            chunk_idx += 1
            print(f"Starting chunk {chunk_idx}...")
            out_f = open(f"i24_chunk_{chunk_idx:02d}.json", "w")
            out_f.write("[\n")
            current_size = 0
            first_item = True

        if not first_item:
            out_f.write(",\n")

        out_f.write(item_str)
        current_size += item_size
        first_item = False

out_f.write("\n]\n")
out_f.close()
print("Splitting complete!")

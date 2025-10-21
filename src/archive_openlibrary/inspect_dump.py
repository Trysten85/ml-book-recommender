"""
Inspect the Open Library dump to see its format
"""
import gzip

filepath = 'data/raw/ol_dump_works_latest.txt.gz'

print("Looking at first 20 lines of the dump:\n")

with gzip.open(filepath, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        
        print(f"--- Line {i} ---")
        print(line[:500])  # First 500 chars
        print()

print("\n\nNow looking for a line that contains actual book data:")

with gzip.open(filepath, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i > 1000:
            break
        
        # Look for lines that contain "title"
        if '"title"' in line and not line.startswith('/type/'):
            print(f"\n--- Found book data at line {i} ---")
            print(line[:1000])
            break
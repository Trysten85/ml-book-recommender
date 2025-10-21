"""
Check the status of the Goodreads enrichment process
Run this anytime to see progress
"""
import os
import sys

# Check if enriched file exists
enriched_file = "data/processed/books_goodreads_enriched.pkl"

if os.path.exists(enriched_file):
    # Get file modification time
    import datetime
    mod_time = os.path.getmtime(enriched_file)
    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    # Get file size
    file_size_mb = os.path.getsize(enriched_file) / (1024 * 1024)

    print("="*60)
    print("ENRICHMENT COMPLETE!")
    print("="*60)
    print(f"Enriched file: {enriched_file}")
    print(f"Last modified: {mod_time_str}")
    print(f"File size: {file_size_mb:.1f} MB")
    print()
    print("To view results:")
    print("  python -c \"import pandas as pd; df = pd.read_pickle('data/processed/books_goodreads_enriched.pkl'); print(f'Total books: {len(df):,}'); print(f'Enriched from Goodreads: {(df[\\\"enrichment_source\\\"] == \\\"goodreads\\\").sum():,}')\"")
    print()
    print("Next steps:")
    print("1. Rebuild model: python src/recommender.py")
    print("2. Test recommendations: python src/test_recommender.py")
else:
    print("="*60)
    print("ENRICHMENT STILL RUNNING...")
    print("="*60)
    print(f"Looking for: {enriched_file}")
    print("File not found yet - enrichment still in progress.")
    print()
    print("The process will create this file when complete.")
    print("Check back in a few minutes...")

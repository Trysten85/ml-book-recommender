# Migration to Goodreads Dataset

**Date**: October 2025
**Status**: ✅ Complete

## Overview

Migrated the book recommendation system from using OpenLibrary (14.5M books) to Goodreads (10K curated books) as the primary dataset.

## Why?

### Problems with OpenLibrary Approach
- **Scale Issues**: 14.5M books caused:
  - 3-4 hour processing times
  - Memory crashes during fuzzy matching
  - 3-4GB dataset files
  - 2GB+ model files
- **Data Quality**:
  - Only ~60% had usable descriptions
  - Inconsistent metadata
  - Required complex API enrichment
- **Complexity**:
  - Multiple enrichment steps (Goodreads API, OpenLibrary API)
  - Fuzzy matching across millions of records
  - Complex error handling

### Benefits of Goodreads Approach
- **Performance**:
  - 5 second processing (vs 3-4 hours)
  - 12MB dataset (vs 4.2GB)
  - 7MB model (vs 2GB)
- **Quality**:
  - 99.4% have descriptions
  - 100% have genres and authors
  - Average 54K ratings per book
- **Simplicity**:
  - Single clean CSV input
  - No API calls needed
  - No fuzzy matching required

## What Changed

### New Files
- `src/process_goodreads.py` - Processes Goodreads CSV into recommender format
- `data/processed/books_goodreads.pkl` - 10K curated books (12MB)
- `data/processed/recommender_model_goodreads.pkl` - New model (7.4MB)
- `README.md` - Complete documentation
- `MIGRATION.md` - This file

### Updated Files
- `src/recommender.py`:
  - Updated default paths to Goodreads dataset
  - Lowered similarity threshold (0.5 → 0.15) for better coverage
  - More lenient rating filter (input - 0.8)
  - Added rating column to output
  - Simplified model loading (removed v2/v3/v4 fallbacks)

- `src/test_recommender.py`:
  - Updated to use exact titles from dataset
  - Added comprehensive test cases
  - Better error handling

### Archived Files

**Location**: `data/archive_openlibrary/` and `src/archive_openlibrary/`

#### Datasets (~16GB)
- `books.pkl` (3.3GB) - Original OpenLibrary dump
- `books_backup_full.pkl` (3.3GB) - Backup
- `books_enriched_ol.pkl` (3.4GB) - OpenLibrary API enriched
- `books_goodreads_enriched.pkl` (4.2GB) - Mixed OL + Goodreads
- `books_quality_en.pkl` (601MB) - Filtered English books
- `authors.pkl` (638MB) - Author data

#### Models (~5GB)
- `recommender_model.pkl` (680KB) - v1
- `recommender_model_v2.pkl` (1.2GB) - v2
- `recommender_model_v3.pkl` (2.1GB) - v3 with subjects
- `recommender_model_v4_enriched.pkl` (2.1GB) - v4 with Goodreads enrichment

#### Scripts
- `enrich_from_openlibrary.py` - OpenLibrary API enrichment
- `enrich_from_goodreads.py` - Goodreads enrichment for OL data
- `add_authors.py` - Author processing
- `add_language_column.py` - Language detection
- `clean_existing_subjects.py` - Subject cleanup
- `filter_quality_books.py` - Quality filtering
- `process_authors.py` - Author extraction
- `inspect_dump.py` - Dump inspection
- `download_data.py` - Data download

### Kept Files (Still Used)
- `process_dump.py` - Contains `detectLanguage()` function used by new system
- `vectorizer.py` - May be useful for future enhancements

## Results

### Performance Comparison

| Metric | Before (OpenLibrary) | After (Goodreads) | Improvement |
|--------|---------------------|-------------------|-------------|
| Dataset Size | 4.2GB | 12MB | **350x smaller** |
| Model Size | 2.1GB | 7.4MB | **284x smaller** |
| Processing Time | 3-4 hours | 5 seconds | **2,160x faster** |
| Build Time | 10+ minutes | 5 seconds | **120x faster** |
| Books | 14.5M | 10K | Focused quality |
| With Descriptions | ~60% | 99.4% | Better coverage |
| With Ratings | ~30% | 100% | Full metadata |

### Test Results

All tests pass successfully:

✅ **Harry Potter**: Gets other HP books + similar fantasy
✅ **The Martian**: Gets sci-fi survival books (Foundation, Alien)
✅ **Golden Son**: Gets same series + similar dystopian fantasy
✅ **The Hunger Games**: Gets HG series + Divergent, dystopian YA

## Disk Space Saved

**Total**: ~21GB → ~19MB = **99.9% reduction**

- Datasets: 15GB → 12MB
- Models: 5.4GB → 7.4MB
- Can delete archived files if space needed

## Trade-offs

### Accepted Limitations
- ❌ Only 10K books (vs 14.5M)
- ❌ Only popular books
- ❌ Primarily English (97.4%)

### Benefits Gained
- ✅ High-quality recommendations
- ✅ Fast and reliable
- ✅ No crashes
- ✅ Easy to maintain
- ✅ Better for portfolio demo

## Cleanup Options

If you need to free up disk space:

```bash
# Delete archived datasets (15GB)
rm -rf data/archive_openlibrary/*.pkl

# Delete archived scripts (keep for reference if needed)
rm -rf src/archive_openlibrary/

# Or delete entire archive (20GB)
rm -rf data/archive_openlibrary/ src/archive_openlibrary/
```

## Next Steps

- [ ] Add fuzzy title search
- [ ] Build web interface
- [ ] Consider expanding to 50K books if needed
- [ ] Add collaborative filtering
- [ ] Deploy to cloud

## Conclusion

The migration was **highly successful**. The system is now:
- ✅ 99.9% smaller
- ✅ 2000x+ faster
- ✅ More reliable
- ✅ Better quality recommendations
- ✅ Easier to maintain

**Recommendation**: Delete archived files once comfortable with new system.
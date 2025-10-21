# Project Status

**Last Updated**: October 2025
**Status**: ✅ Production Ready - To be Enhanced with User History Support

## Current State

The Book Recommender System has been successfully migrated from OpenLibrary to Goodreads as the primary data source. Recent enhancements include:

- **Length-based similarity adjustment** for better series book recommendations
- **Fuzzy book search** for easier book lookup
- **Batch recommendations** from user reading history (2 strategies)
- **Code cleanup** - freed 12GB disk space, archived test scripts

### Active Files (32MB total)

```
Book Recomender/
├── src/                            # 4 Python files
│   ├── recommender.py              # Main recommendation engine (NEW FEATURES!)
│   ├── test_recommender.py         # Test suite with examples
│   ├── process_goodreads.py        # Dataset processor
│   └── process_dump.py             # Utilities (detectLanguage)
├── data/
│   ├── external/
│   │   └── goodbooks_10k_enriched.csv     (10MB)
│   └── processed/
│       ├── books_goodreads.pkl            (12MB)
│       ├── books_goodreads.csv            (13MB)
│       └── recommender_model_goodreads.pkl (7.4MB)
├── scripts/
│   └── archive_tests/              # 10 archived test scripts
├── README.md                       # Complete documentation
├── MIGRATION.md                    # Migration details
└── STATUS.md                       # This file
```

### Recent Cleanup (Freed 12GB)

**October 2025 Cleanup:**
- Moved 5 old CSV files (9GB) to `data/archive_openlibrary/`
  - authors.csv, books.csv, books_enriched_ol.csv, etc.
- Deleted `data/processed/temp_chunks/` (3.4GB of processing artifacts)
- Archived 10 test scripts to `scripts/archive_tests/`
- Removed empty `src/vectorizer.py`

**Still Archived:**
- `data/archive_openlibrary/` (30GB+ with cleanup additions)
- Can be safely deleted after verifying system works

## System Overview

### Dataset
- **Source**: Goodreads 10K
- **Books**: 10,000 curated popular titles
- **Quality**: 99.4% have descriptions, 100% have genres/authors
- **Size**: 12MB (vs 4.2GB previously)
- **Language**: 97.4% English

### Model
- **Algorithm**: TF-IDF + Cosine Similarity
- **Features**: 3000 (descriptions + 3x weighted genres)
- **Length Penalty**: 0.5 (soft penalty for similar description lengths)
- **Size**: 7.4MB (vs 2.1GB previously)
- **Build Time**: 5 seconds
- **Accuracy**: High quality recommendations with improved series matching

### Performance
- **Dataset Loading**: <1 second
- **Model Loading**: <1 second
- **Recommendation Query**: <0.1 seconds
- **Memory Usage**: ~500MB (vs 4GB+ previously)

## Test Results

✅ All tests passing:

1. **Harry Potter and the Prisoner of Azkaban**
   - Returns: Other HP books + similar children's fantasy
   - Similarity: 0.45-0.59
   - Quality: Excellent

2. **The Martian**
   - Returns: Foundation, Alien, The Long Walk
   - Similarity: 0.26-0.34
   - Quality: Great sci-fi matches

3. **Golden Son (Red Rising series)**
   - Returns: Morning Star (same series) + fantasy
   - Similarity: 0.39-0.51
   - Quality: Excellent

4. **The Hunger Games**
   - Returns: Mockingjay, Catching Fire, Divergent
   - Similarity: 0.36-0.60
   - Quality: Perfect YA dystopian matches

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Books | 10,000 |
| With Descriptions | 9,943 (99.4%) |
| With Genres | 10,000 (100%) |
| With Ratings | 10,000 (100%) |
| Average Rating | 4.0 / 5.0 |
| Dataset Size | 12MB |
| Model Size | 7.4MB |
| Processing Time | 5 seconds |
| Recommendation Speed | <0.1s |

## API

### Initialize
```python
from recommender import BookRecommender

recommender = BookRecommender()
recommender.loadModel()
```

### Single Book Recommendations
```python
recs = recommender.getRecommendations(
    bookTitle="The Hunger Games (The Hunger Games, #1)",
    bookAuthor="Suzanne Collins",
    n=10,
    filterLanguage=True,
    minDescriptionLength=50,
    minRating=None  # Auto-adjusts based on input
)
```

### NEW: Fuzzy Book Search
```python
# Find books by partial title match
results = recommender.findBookByTitle("hunger", n=5)
# Returns: All books with "hunger" in title
```

### NEW: Batch Recommendations (User History)
```python
# Get recommendations based on multiple books user has read
recs = recommender.getRecommendationsFromHistory(
    bookTitles=[
        "The Hunger Games (The Hunger Games, #1)",
        "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",
        "The Martian"
    ],
    bookAuthors=["Suzanne Collins", "J.K. Rowling", "Andy Weir"],
    n=10,
    strategy='aggregate'  # 'aggregate' or 'top_per_book'
)
```

### Output Format
DataFrame with columns:
- `title`: Book title
- `authorNames`: Author(s)
- `subjects`: Genres/subjects
- `similarityScore`: 0.0-1.0
- `average_rating`: Goodreads rating
- `detectedLanguage`: Language code

## Known Limitations

1. **Coverage**: Only 10,000 books (popular titles only)
2. **Language**: Primarily English (97.4%)
3. **Method**: Content-based only (no collaborative filtering yet)

## Recent Enhancements

### ✅ Completed (October 2025)
- [x] Length-based similarity adjustment (improves series book matching)
- [x] Fuzzy title matching for easier lookups
- [x] Batch recommendations from user history (2 strategies)
- [x] Code cleanup - freed 12GB disk space

### Future Enhancements

### Short Term
- [ ] Create simple web UI (Flask/Streamlit)
- [ ] Add user profile storage (JSON/SQLite)
- [ ] Export recommendations to CSV/PDF

### Medium Term
- [ ] Expand to 50K books dataset
- [ ] Add collaborative filtering
- [ ] Book rating prediction
- [ ] Genre-specific models

### Long Term
- [ ] Deploy as web service (Heroku/Railway)
- [ ] User accounts with authentication
- [ ] Mobile-responsive interface
- [ ] Integration with Goodreads API for live data

## Usage Examples

### Command Line
```bash
# Run tests
python src/test_recommender.py

# Process dataset (if updating)
python src/process_goodreads.py

# Rebuild model (if needed)
python src/recommender.py
```

### Python Script
```python
from recommender import BookRecommender

# Initialize
rec = BookRecommender()
rec.loadModel()

# Get recommendations
results = rec.getRecommendations(
    bookTitle="The Martian",
    bookAuthor="Andy Weir",
    n=5
)

# Display
print(results[['title', 'authorNames', 'similarityScore']])
```

## Maintenance

### Updating Dataset
1. Get new Goodreads CSV
2. Run: `python src/process_goodreads.py`
3. Rebuild: `python src/recommender.py`
4. Test: `python src/test_recommender.py`

### Troubleshooting

**"Book not found"**
- Check exact title in dataset
- Use partial author name (e.g., "Rowling" not "J.K. Rowling")

**"No recommendations found"**
- Lower `minDescriptionLength` parameter
- Disable `filterLanguage` if testing non-English
- Check book has sufficient description/genres

**Memory issues**
- Shouldn't happen with 10K dataset
- If it does, reduce `max_features` in TF-IDF

## Clean Up Archive

After confirming system works, free up 21GB:

```bash
# Delete old datasets and models
rm -rf data/archive_openlibrary/

# Delete old scripts (keep for reference if unsure)
rm -rf src/archive_openlibrary/
```

## Documentation

- **README.md**: User guide and technical details
- **MIGRATION.md**: Migration from OpenLibrary details
- **STATUS.md**: This file - current project state

## Contact

For questions or issues:
1. Check README.md
2. Check MIGRATION.md for historical context
3. Review test_recommender.py for examples
4. Create GitHub issue (if repository is public)

---

**System Status**: ✅ Operational
**Last Test**: Successful
**Ready for**: Portfolio, Demo, Further Development
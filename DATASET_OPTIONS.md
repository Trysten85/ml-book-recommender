# Dataset Expansion Options

Your current dataset: **9,768 high-quality books from Goodreads**

Goal: Expand to **50,000-60,000 books**

## Option 1: Kaggle Goodreads Dataset (RECOMMENDED)

**Dataset:** https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m
- **Size:** 2M+ books, updated every 2 days
- **Fields:** Same as your current dataset (title, author, description, ratings, etc.)
- **Quality:** High - from Goodreads

**How to download:**

### Manual Method (Easiest):
1. Go to: https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m
2. Click "Download" button (requires free Kaggle account)
3. Extract ZIP to: `data/kaggle/`
4. Run: `python src/process_external_books.py --source kaggle`

### API Method:
```bash
# Install Kaggle API
pip install kaggle

# Setup credentials (one-time)
# 1. Go to kaggle.com -> Account -> Create New API Token
# 2. Place kaggle.json in C:\Users\Trysten\.kaggle\

# Download
python src/download_kaggle_books.py
```

---

## Option 2: Goodreads Best Books Dataset

**Dataset:** https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
- **Size:** 11,000 books
- **Quality:** Very high - all popular/best books
- **Smaller but cleaner**

**How to download:**
1. Visit link above
2. Download ZIP
3. Extract to: `data/external/`
4. Run: `python src/process_external_books.py --source goodreads-best`

---

## Option 3: Amazon Books Dataset

**Dataset:** https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
- **Size:** 212,000+ books
- **Fields:** Title, author, description, ratings, reviews, price

**How to download:**
1. Visit link above
2. Download ZIP
3. Extract to: `data/external/`
4. Run: `python src/process_external_books.py --source amazon`

---

## Processing Pipeline (After Download)

Once you download ANY of these datasets, I've created a processing script that will:

1. **Load external dataset**
2. **Apply aggressive filters:**
   - English only (language detection)
   - Description ≥ 100 characters
   - Has author, genres, ratings
   - Ratings count ≥ 100
   - Remove companion books (cookbook, guide, etc.)
   - Remove duplicates

3. **Merge with your existing 9,768 books**
4. **Deduplicate** (by ISBN/title+author)
5. **Generate embeddings** (optimized for CPU)
6. **Save** to `data/processed/books_expanded.pkl`

**Expected output:** 50,000-60,000 high-quality books ready for training

---

## Which Option Should You Choose?

**Best for maximum books:** Option 1 (Kaggle 10M - filter to 50k-60k)
**Best for quality:** Option 2 (Goodreads Best - 11k premium books)
**Best for variety:** Option 3 (Amazon - 200k+ books)

**My recommendation:** Try Option 1 first (Kaggle 10M). If download is slow, go with Option 2 (smaller, faster).

---

## Next Steps

1. **Choose a dataset** and download it
2. Let me know which one you downloaded
3. I'll run the processing script to filter, merge, and prepare for training

**Questions?** Just ask!

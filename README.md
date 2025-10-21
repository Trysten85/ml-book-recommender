# Book Recommender - ML-Powered Book Discovery

> An intelligent book recommendation system powered by semantic embeddings, featuring 109,632 curated English books.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

## Features

- **Semantic Search**: Find books using natural language queries
- **AI Recommendations**: Get similar books based on 384-dimensional semantic embeddings
- **Clean Dataset**: 109,632 English books (foreign content, duplicates removed)
- **Multi-User Support**: Personal libraries with Read/Reading/Want to Read shelves
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Mobile-Responsive UI**: React frontend (in development)

## Tech Stack

**Backend**
- FastAPI - Modern Python web framework
- scikit-learn - ML and similarity calculations
- SentenceTransformers - Semantic embeddings (all-MiniLM-L6-v2)
- Pandas - Data processing

**Frontend** (In Development)
- React + Vite - Fast, modern UI
- Tailwind CSS - Mobile-first styling
- Axios - API client

**Data**
- Source: Kaggle Goodreads dataset
- Size: 109,632 books (cleaned from 152K)
- Embeddings: 384-dimensional vectors

## Quick Start

### 1. Download Large Data Files

Large datasets (>2GB) are stored separately on AWS S3:

```bash
# Download from S3 bucket
aws s3 sync s3://ml-book-recommender-data ./data --exclude "*" --include "goodreads/*"

# Or download manually from:
# s3://ml-book-recommender-data
```

**Required files:**
- `data/goodreads/goodreads_books.json` (large book metadata)
- `data/goodreads/goodreads_interactions.csv` (user interactions)

### 2. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python api/main.py
```

API runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

## Project Structure

```
Book Recomender/
├── api/                      # FastAPI backend
│   └── main.py              # API endpoints
├── src/                      # Core ML engine
│   ├── recommender.py       # BookRecommender class
│   ├── user_library.py      # User library management
│   └── user_manager.py      # Multi-user support
├── scripts/                  # Data processing
│   ├── clean_complete.py    # Dataset cleaning pipeline
│   └── filter_embeddings.py # Embedding filtering
├── data/
│   ├── raw/                 # Original Kaggle data
│   └── processed/           # Cleaned data + embeddings
│       ├── books_clean.pkl  # 109K cleaned books
│       └── books_clean_embeddings.npy # Semantic vectors
├── tests/                    # API test suite
│   └── test_api_endpoints.py
├── frontend/                 # React UI (coming soon)
└── archive/                  # Old scripts/notebooks
```

## API Endpoints

### Core Endpoints
- `POST /api/search` - Search books with pagination
- `POST /api/search-book-for-recommendations` - Find book for recommendations
- `POST /api/recommendations` - Get similar books by ISBN
- `GET /api/book/{isbn13}` - Get book details

### User Management
- `GET /api/users` - List all users
- `POST /api/users` - Create new user
- `GET /api/users/{username}/library` - Get user's library
- `POST /api/users/{username}/library/add` - Add book to shelf
- `POST /api/users/{username}/library/recommendations` - Personalized recommendations

## Dataset Pipeline

### Cleaning Process
The cleaning pipeline (`scripts/clean_complete.py`) performs:

1. **Foreign Language Removal**: Filters non-English books by title patterns
2. **Deduplication**: Merges duplicate editions, keeps best description/cover
3. **Non-Novel Filtering**: Removes box sets, companion books, audiobooks
4. **Quality Filters**: Minimum description length, removes marketing text

**Result**: 152,650 → 109,632 books (27.2% reduction)

### Embeddings
- Model: all-MiniLM-L6-v2 (SentenceTransformers)
- Input: Book descriptions + subjects
- Output: 384-dimensional semantic vectors
- Use: Cosine similarity for recommendations

## Recommendation Algorithm

### Two-Step Workflow
1. **Search** → Get list of matching books
2. **Select** → Get recommendations by ISBN

### Similarity Calculation
- Uses semantic embeddings (cosine similarity)
- Filters by language, rating, description quality
- Excludes books in same series (except next book)
- Adaptive rating threshold (based on input book)

## Testing

Run API test suite:

```bash
python tests/test_api_endpoints.py
```

**Results**: 6/8 tests passing (75%)
- All core functionality working
- Minor issues: duplicate handling, ISBN format edge cases

## Development Status

- ✅ Dataset cleaning and processing
- ✅ ML model (semantic embeddings)
- ✅ FastAPI backend with full endpoints
- ✅ Multi-user library system
- ✅ API testing suite
- 🚧 React frontend (in progress)
- 📋 Deployment (planned)

## Performance

- **Dataset**: 109,632 books
- **Model Size**: ~42MB (embeddings)
- **Recommendation Speed**: < 0.2 seconds
- **API Response Time**: < 0.5 seconds

## Future Enhancements

- [ ] Complete React frontend
- [ ] User authentication (JWT)
- [ ] Advanced filters (genre, year, rating)
- [ ] Social features (share recommendations)
- [ ] PWA support (offline mode)
- [ ] Deploy to production

## Contributing

This is a portfolio project. Feedback and suggestions welcome!

## License

MIT - Built for educational/portfolio purposes.

## Author

Portfolio project showcasing full-stack ML application development.

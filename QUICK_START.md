# Quick Start Guide

## 🚀 How to Run the Application

### Step 1: Start the Backend API

Open a terminal and run:

```bash
cd "c:\Users\Trysten\Documents\Projects\Book Recomender"
python api/main.py
```

You should see:
```
✓ Loaded 109,632 books (English only, collections excluded)
Uvicorn running on http://0.0.0.0:8000
```

**API is now running at:** `http://localhost:8000`
**API Docs:** `http://localhost:8000/docs`

---

### Step 2: Start the Frontend

Open a **NEW terminal** and run:

```bash
cd "c:\Users\Trysten\Documents\Projects\Book Recomender\frontend"
npm run dev
```

You should see:
```
VITE v7.1.11  ready in XXX ms
➜  Local:   http://localhost:5173/
```

**Frontend is now running at:** `http://localhost:5175` (or 5174, 5175 if ports are in use)

---

### Step 3: Open in Browser

Open your web browser and go to the URL shown in the terminal (usually):
```
http://localhost:5175
```
(If port 5173 is in use, Vite will automatically use 5174, 5175, etc.)

You should see the Book Recommender homepage with a search bar!

---

## 🎯 How to Use the App

1. **Search for a book**
   - Type "Harry Potter" in the search bar
   - Wait for search results to appear

2. **Get recommendations**
   - Click "Find Similar Books" on any book card
   - See AI-powered recommendations with similarity scores

3. **Try different searches**
   - "The Hobbit"
   - "1984"
   - "Brandon Sanderson"
   - Any book or author you like!

---

## ⏹️ How to Stop the Servers

### Stop Backend (API)
In the terminal where API is running:
- Press `Ctrl + C`

### Stop Frontend
In the terminal where frontend is running:
- Press `Ctrl + C`

---

## 🛠️ Troubleshooting

### Port 8000 already in use?
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace XXXX with PID from above)
taskkill /F /PID XXXX
```

### Port 5173 already in use?
```bash
# Find process using port 5173
netstat -ano | findstr :5173

# Kill the process (replace XXXX with PID from above)
taskkill /F /PID XXXX
```

### Frontend shows blank page?
- Make sure backend API is running first
- Check browser console for errors (F12)
- Verify API is accessible at `http://localhost:8000/docs`

### Search returns no results?
- Make sure you're searching for popular books
- API only has English books
- Try: "Harry Potter", "Lord of the Rings", "The Hobbit"

---

## 📝 Quick Commands

```bash
# Backend API
cd "c:\Users\Trysten\Documents\Projects\Book Recomender"
python api/main.py

# Frontend (in a new terminal)
cd "c:\Users\Trysten\Documents\Projects\Book Recomender\frontend"
npm run dev

# Run API tests
python tests/test_api_endpoints.py
```

---

## 🌐 URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | React UI |
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Interactive API documentation |

---

## 💡 Tips

- Keep both terminals open while using the app
- Backend must be running for frontend to work
- Changes to frontend code will auto-reload (hot reload)
- Changes to backend code require restarting the server
- Use Ctrl+C to stop servers gracefully

---

**Ready to start?** Open two terminals and follow Steps 1 & 2 above! 🚀

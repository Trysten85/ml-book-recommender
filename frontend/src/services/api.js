import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Book Search & Discovery
export const searchBooks = async (query, page = 1, perPage = 20) => {
  const response = await api.post('/api/search', {
    query,
    page,
    per_page: perPage,
  });
  return response.data;
};

export const searchBooksForRecommendations = async (query) => {
  const response = await api.post('/api/search-book-for-recommendations', {
    query,
  });
  return response.data;
};

export const getRecommendations = async (isbn13, n = 10) => {
  const response = await api.post('/api/recommendations', {
    isbn13,
    n,
  });
  return response.data;
};

export const getBookDetails = async (isbn13) => {
  const response = await api.get(`/api/book/${isbn13}`);
  return response.data;
};

// User Management
export const createUser = async (username) => {
  const response = await api.post('/api/users', { username });
  return response.data;
};

export const getUserLibrary = async (username) => {
  const response = await api.get(`/api/users/${username}/library`);
  return response.data;
};

export const addBookToLibrary = async (username, bookData) => {
  const response = await api.post(`/api/users/${username}/library/add`, bookData);
  return response.data;
};

export const getUserRecommendations = async (username, n = 10) => {
  const response = await api.post(`/api/users/${username}/library/recommendations`, { n });
  return response.data;
};

export default api;

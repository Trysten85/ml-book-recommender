import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import RecommendationsPage from './pages/RecommendationsPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/recommendations/:isbn" element={<RecommendationsPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import Upload from './components/Upload';
import Results from './components/Results';
import Predict from './components/Predict';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/results" element={<Results />} />
                <Route path="/predict" element={<Predict />} />
            </Routes>
        </Router>
    );
};

export default App;

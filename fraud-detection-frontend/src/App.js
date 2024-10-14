import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import Upload from './components/Upload';
import Results from './components/Results';
import Predict from './components/Predict';

const App = () => {
    return (
        <Router>
            <Switch>
                <Route path="/" exact component={LandingPage} />
                <Route path="/upload" component={Upload} />
                <Route path="/results" component={Results} />
                <Route path="/predict" component={Predict} />
            </Switch>
        </Router>
    );
};

export default App;

import React, { useState, useEffect } from 'react';
import { backend } from './services/backendBridge';
import PosteriorVisualization from './components/PosteriorVisualization';
import ExperienceControls from './components/ExperienceControls';
import MeditationControls from './components/MeditationControls';
import SufferingMetrics from './components/SufferingMetrics';
import './App.css';

function App() {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchState = async () => {
    try {
      const data = await backend.getState();
      setState(data);
    } catch (error) {
      console.error('Error fetching state:', error);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchState();

    // Poll state every 200ms (5 times per second) for smooth decay animation
    const interval = setInterval(() => {
      fetchState();
    }, 200);

    return () => clearInterval(interval);
  }, []);

  const handleAddExperience = async (value, sigma = 1.0) => {
    setLoading(true);
    try {
      await backend.addExperience(value, sigma);
      await fetchState();
    } catch (error) {
      console.error('Error adding experience:', error);
    }
    setLoading(false);
  };

  const handleMeditate = async (meditationLevel) => {
    try {
      await backend.meditate(meditationLevel);
      await fetchState();
    } catch (error) {
      console.error('Error meditating:', error);
    }
  };

  const handleSetBrahmaViharas = async (level) => {
    try {
      await backend.setBrahmaViharas(level);
      await fetchState();
    } catch (error) {
      console.error('Error setting brahma viharas:', error);
    }
  };

  const handleReset = async () => {
    try {
      await backend.reset();
      await fetchState();
    } catch (error) {
      console.error('Error resetting:', error);
    }
  };

  if (!state) {
    return (
      <div className="app loading">
        <div className="loading-spinner">Loading Dharma Visualizer...</div>
        <div className="loading-subtitle">Initializing Python Environment (Pyodide)</div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧘 Dharma Visualizer</h1>
        <p className="subtitle">
          Bayesian Belief Formation & Suffering Through Precision-Weighted Inference
        </p>
        <button className="reset-button" onClick={handleReset}>
          Reset Model
        </button>
      </header>

      <div className="main-content">
        <div className="left-panel">
          <section className="card">
            <h2>Add Experience</h2>
            <ExperienceControls
              onAddExperience={handleAddExperience}
              loading={loading}
            />
          </section>

          <section className="card">
            <h2>Meditation & Cultivation</h2>
            <MeditationControls
              experiences={state.experiences}
              onMeditate={handleMeditate}
              brahmaViharasLevel={state.brahma_viharas_level || 0}
              onSetBrahmaViharas={handleSetBrahmaViharas}
            />
          </section>
        </div>

        <div className="right-panel">
          <section className="card visualization">
            <h2>Posterior Distribution & Experiences</h2>
            <PosteriorVisualization
              posterior={state.posterior}
              experienceDistributions={state.experience_distributions}
              experiences={state.experiences}
            />
          </section>

          <section className="card">
            <h2>Suffering Metrics</h2>
            <SufferingMetrics metrics={state.suffering_metrics} />
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;

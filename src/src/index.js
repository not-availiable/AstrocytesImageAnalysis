import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import '@fontsource/montserrat/300.css';  // Light
import '@fontsource/montserrat/400.css';  // Regular
import '@fontsource/montserrat/500.css';  // Medium
import '@fontsource/montserrat/700.css';  // Bold

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
// Importar Bootstrap CSS aquí asegura que esté disponible globalmente
import 'bootstrap/dist/css/bootstrap.min.css';
// Importar tus estilos personalizados DESPUÉS de bootstrap para que puedan sobreescribir
import './styles/App.css';

const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
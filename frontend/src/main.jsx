import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import TestPage from './components/TestPage'
import './styles/global.css'

const isTestPage = window.location.pathname === '/test'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {isTestPage ? <TestPage /> : <App />}
  </React.StrictMode>
)

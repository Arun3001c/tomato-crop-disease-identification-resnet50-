import { useState } from 'react'
import axios from 'axios'
import './App.css'  // Import the CSS file here

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setPrediction('')
      setError('')
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    setPreview(null) // Hide preview when predicting
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      const data = response.data
      setPrediction(data.prediction + (data.confidence ? ` (Confidence: ${(data.confidence * 100).toFixed(2)}%)` : ''))
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="boy">
      <div className="container">
        <div className="card">
          <h1 className="title">🌿 Plant Disease Detector</h1>
          <p className="subtitle">Upload a leaf image to detect disease</p>

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="fileInput"
          />

          {preview && (
            <div className="imageWrapper">
              <img src={preview} alt="Preview" className="image" />
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className={`button ${loading ? 'buttonLoading' : ''}`}
          >
            {loading ? '🔄 Predicting...' : '🔍 Predict'}
          </button>

          {prediction && (
            <div className="resultBox resultSuccess">
              <strong>Prediction:</strong> {prediction}
            </div>
          )}

          {error && (
            <div className="resultBox resultError">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
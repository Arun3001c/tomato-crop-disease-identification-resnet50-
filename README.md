# Plant Disease Prediction App

This project consists of a backend API and a frontend web application for predicting plant diseases using a ResNet50 model.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Backend Setup

1. Navigate to the `backend` directory:
   ```
   cd backend
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```
   python app.py
   ```

   The backend will start on `http://localhost:5000` (or the port specified in `app.py`).

## Frontend Setup

1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```

2. Install Node.js dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

   The frontend will start on `http://localhost:5173` (default Vite port).

## Running the Application

- Start the backend first, then the frontend.
- Open your browser and go to the frontend URL to use the application.
- The frontend communicates with the backend API for disease predictions.
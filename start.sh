#!/bin/bash

# To run this script, navigate to the root directory of the project and run the following command:
# chmod +x start.sh
# ./start.sh

# Navigate to backend directory and start FastAPI server
echo "Starting Flask server..."
cd backend
python server.py &  

# Navigate to frontend directory and start React app
echo "Starting Vue frontend..."
cd frontend
npm run serve &  

wait # Wait for both processes to finish
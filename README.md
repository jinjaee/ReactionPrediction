EOH Reaction Prediction Engine
A machine-learning powered engine that predicts the stable products of inorganic chemical reactions. It uses a Random Forest model trained on 132,000 materials to calculate thermodynamic stability (Convex Hull) and determine reaction outcomes.

🚀 Quick Start
1. Clone the Repository
Open your terminal and download the project:

Bash

git clone https://github.com/jinjaee/ReactionPrediction.git


cd ReactionPrediction


2. Install Dependencies


Install the required Python libraries for the AI and Server:

Bash

pip install fastapi uvicorn pandas numpy scikit-learn pymatgen matminer joblib


3. Initialize the "Brain"
Note: The trained model files (.pkl) are too large for GitHub, so you must generate them once on your machine. This takes about 5–10 minutes.

Run the training script to build the model locally:

Bash

python src/train_rf.py
Wait until you see "Success! Model saved to models/rf_model.pkl".

🖥️ Running the Application
You need two terminal windows open to run the full stack.

Terminal 1: Start the Backend (API)
This runs the AI engine on localhost:8000.

Bash

# Make sure you are in the project folder
python api.py
Terminal 2: Start the Frontend (Website)
This launches the user interface.

Bash

cd frontend
python -m http.server 8080
Use the App
Open your web browser and go to: 👉 http://localhost:8080

Click "Mg" and "O".

Watch the AI predict MgO!

🛠️ Tech Stack
Backend: Python, FastAPI, Scikit-Learn

Chemistry: Pymatgen, Matminer

Frontend: HTML5, CSS3, Vanilla JavaScript

Algorithm: Random Forest Regressor (Thermodynamic Stability Prediction)
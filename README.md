\# ⚡ EnerVision – AI Energy Forecasting for Indian Buildings



\## 🌱 About

EnerVision is an AI-driven prototype that forecasts energy usage, detects anomalies, and recommends efficiency measures.  

Built during the \*\*IndiaAI Impact Gen-AI Hackathon\*\*, it addresses India’s urgent need for intelligent building energy management tools.



---



\## 🚀 Features

\- \*\*Load Forecasting\*\* → Predict 24h–7d demand using TSFMs \& Prophet.

\- \*\*Anomaly Detection\*\* → Identify unusual usage (e.g., faulty appliances).

\- \*\*Advisor Agent\*\* → Suggest off-peak usage, solar, and battery adoption.

\- \*\*Dashboard\*\* → React-based UI for forecasts, anomalies, and advice.



---



\## 🛠️ Tech Stack

\- \*\*Backend\*\*: FastAPI, PyTorch, Prophet

\- \*\*Frontend\*\*: React, Streamlit (prototype)

\- \*\*Databases\*\*: PostgreSQL, Redis

\- \*\*Containerization\*\*: Docker, docker-compose

\- \*\*Planned Deployment\*\*: IBM watsonx.ai

\- \*\*Models\*\*: TSFMs, Isolation Forest, Autoencoders



---



\## 📂 Repo Structure

EnerVision/

│── backend/ # FastAPI + AI + DB

│── frontend/ # React dashboard

│── data/ # Sample datasets

│── docs/ # Architecture, roadmap, notes

│── docker-compose.yml # Orchestration

│── .gitignore

│── README.md

\## 🏃 Getting Started



\### Clone the repo

```bash

git clone https://github.com/<your-username>/EnerVision.git

cd EnerVision



Run with Docker

docker-compose up --build



Backend → http://localhost:8000



Frontend → http://localhost:3000



✨ EnerVision is a prototype today, but a scalable AI solution tomorrow for smarter, greener buildings.




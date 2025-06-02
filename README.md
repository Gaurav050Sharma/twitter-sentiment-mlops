# Twitter Sentiment Analysis MLOps Project

This project implements a Twitter sentiment analysis application with MLOps best practices. The application uses deep learning to classify tweets into four sentiment categories: Neutral, Irrelevant, Negative, and Positive.

## Project Structure

```
twitter_sentiment_mlops/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── data/                  # Data directory
│   └── twitter_training.csv
├── models/                # Model directory
│   └── sentiment_model.h5
└── .github/
    └── workflows/
        └── docker-build.yml  # CI/CD pipeline
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd twitter_sentiment_mlops
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place required files:
- Put your training data in `data/twitter_training.csv`
- Put your trained model in `models/sentiment_model.h5`

5. Run the application locally:
```bash
streamlit run app.py
```

## Docker Support

Build the Docker image:
```bash
docker build -t twitter-sentiment .
```

Run the container:
```bash
docker run -p 8501:8501 twitter-sentiment
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Builds the Docker image
2. Pushes it to Docker Hub

Required secrets:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN

## Model Information

The sentiment analysis model is a deep learning model built with:
- Embedding layer
- Bidirectional LSTM layers
- Dense output layer with softmax activation

The model classifies tweets into four categories:
- Neutral
- Irrelevant
- Negative
- Positive 
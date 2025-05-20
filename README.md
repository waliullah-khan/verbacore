# AI Response Analysis Dashboard

A comprehensive dashboard application for analyzing AI assistant conversations with hallucination detection, topic modeling, sentiment analysis, time series analysis, and text clustering.

## 🌟 [Live Demo](https://ai-review-demo.vercel.app/)

## Features

- **File Upload**: Upload JSON files containing conversation data.
- **Hallucination Detection**: Analyze AI responses for hallucinations using Patronus.
- **Topic Modeling**: Discover main topics in your conversation data.
- **Sentiment Analysis**: Analyze emotional tone in messages.
- **Time Series Analysis**: Explore temporal patterns in message frequency.
- **Text Embeddings & Clustering**: Group similar messages together.
- **Interactive Visualizations**: Explore data through interactive charts and tables.

## Prerequisites

- Python 3.8+
- A Patronus API key (for hallucination detection)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/waliullah-khan/ai_review_demo.git
   cd ai_review_demo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Patronus API key:
   - Create a `.env` file in the project root
   - Add your API key: `PATRONUS_API_KEY=your_api_key_here`
   - Alternatively, set it as an environment variable

## Usage

Run the Streamlit app:

```
streamlit run dashboard_app.py
```

Then open your browser and navigate to http://localhost:8501

## Data Format

The dashboard expects JSON files with a specific structure:

- Must contain a `content` column with message text
- For message pairing, should have a `sender` column (with values 'candidate' or 'bot')
- For time series analysis, should have a `createdAt` column with timestamps
- For threading, should have a `botThreadId` column

Example JSON structure:
```json
[
  {
    "content": "What is the salary range for this position?",
    "sender": "candidate",
    "intent": "candidateAssist",
    "botThreadId": {"$oid": "67aa6a9ce530e06dde29be18"},
    "createdAt": {"$date": "2025-02-10T21:07:40.313Z"}
  },
  {
    "content": "The salary range for this position is $75,000-$95,000 depending on experience.",
    "sender": "bot",
    "intent": "candidateAssist",
    "botThreadId": {"$oid": "67aa6a9ce530e06dde29be18"},
    "createdAt": {"$date": "2025-02-10T21:07:45.123Z"}
  }
]
```

## Deployment

The application is deployed on Vercel:

1. Fork this repository
2. Sign up for a Vercel account at https://vercel.com/
3. Create a new project and import your GitHub repository
4. Add your environment variables in Vercel settings:
   - Add `PATRONUS_API_KEY` with your API key
5. Deploy the application

## Analysis Details

### Topic Modeling
Uses Latent Dirichlet Allocation (LDA) to identify main topics in the conversation data, showing which topics are most prevalent and how they distribute across messages.

### Sentiment Analysis
Analyzes the emotional tone of messages using VADER (Valence Aware Dictionary and sEntiment Reasoner), providing sentiment scores and distribution.

### Time Series Analysis
Examines temporal patterns in message frequency, including daily, weekly, and monthly trends, along with time-of-day distribution.

### Text Clustering
Groups similar messages using TF-IDF vectorization, t-SNE dimensionality reduction, and K-means clustering.

### Hallucination Detection
Uses Patronus to evaluate AI responses for potential hallucinations, providing scores and detailed analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Patronus](https://www.withpatroni.us/) for hallucination detection
- [Streamlit](https://streamlit.io/) for the interactive web application
- [Plotly](https://plotly.com/) for interactive visualizations
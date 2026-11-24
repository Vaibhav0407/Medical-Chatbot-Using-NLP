# AI Medical Chatbot 🩺💬

A web-based chatbot that answers questions about diseases and helps users make sense of their symptoms. Ask about a condition by name and it tells you the type, symptoms, causes, treatment, cure and prevention. Describe your symptoms and it suggests matching conditions.

> **Disclaimer:** This is an academic project for educational use. It does not give medical advice and is not a substitute for a doctor. In an emergency, contact local emergency services.

## What It Does

**Disease lookup.** Type a disease name (for example, "tell me about malaria") and the bot returns a structured summary from its knowledge base.

**Symptom matching.** Type a message that includes the word "symptom" along with what you're feeling. The bot extracts the symptoms and fuzzy-matches them against the disease data.

**Friendly fallbacks.** If it can't understand a message, it says so and tells you what you can ask instead.

## How It Works

1. The user sends a message from the web chat page.
2. Flask receives it at the `/chat` endpoint.
3. If the message contains a known disease name, the bot returns that disease's full profile from `Untitled-1.json`.
4. If the message mentions symptoms, spaCy's NER pipeline extracts them. If no symptom entities are found, the bot falls back to treating the input as a comma-separated list.
5. Extracted symptoms are matched against the disease data using fuzzy string matching (FuzzyWuzzy), so small typos and wording differences still work.
6. The reply is sent back as JSON and shown in the chat window.

The project also includes a DistilBERT question-answering pipeline and a scikit-learn Logistic Regression classifier for experimentation with TF-IDF features.

## Tech Stack

Python, Flask, Flask-CORS, spaCy, Hugging Face Transformers (DistilBERT), scikit-learn, FuzzyWuzzy, HTML/CSS/JavaScript

## Project Structure

```
.
├── app.py              # Flask app and chat endpoint
├── Untitled-1.json     # Disease knowledge base
├── templates/
│   └── index.html      # Chat interface
├── static/             # CSS, JS and assets
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install dependencies**

```bash
pip install flask flask-cors spacy transformers torch scikit-learn fuzzywuzzy
python -m spacy download en_core_web_sm
```

**3. Run the app**

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Example Queries

```
Tell me about diabetes
I have these symptoms: fever, headache, body ache
```

## Evaluation

On an 80/20 train-test split of a labeled symptom-disease dataset, the prototype reported:

| Metric | Score |
|---|---|
| Accuracy | 87% |
| Precision | 85% |
| Recall | 88% |
| F1-Score | 86.5% |

Most errors came from ambiguous wording, very short queries, and over-reliance on single keywords.

## Limitations

This is a prototype. It only knows the diseases in its JSON file, symptom matching is triggered by the word "symptom", and it doesn't remember earlier messages in a conversation. It runs locally and hasn't been tested at scale.

## Roadmap

Multi-turn conversation memory, multilingual support, voice input, a larger and more diverse medical dataset, user feedback collection, cloud deployment, and stronger privacy and security features.

## Contributing

Suggestions and pull requests are welcome. Open an issue if you find a bug or have an idea.

## License

MIT License. See the `LICENSE` file for details.

---

Built by [Vaibhav Gupta](https://github.com/<your-username>)

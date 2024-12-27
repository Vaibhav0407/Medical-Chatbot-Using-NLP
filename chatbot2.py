import json
import spacy
from transformers import pipeline, DistilBertForQuestionAnswering, DistilBertTokenizerFast
from spacy.training import Example
import subprocess
import os
import spacy.cli
from spacy.tokens import DocBin
from spacy.training.example import Example
from fuzzywuzzy import process
from spacy.training import offsets_to_biluo_tags
from spacy.lang.en import English
# Load the trained spaCy model for symptom extraction (Replace "model-best" with your trained model path)
nlp = spacy.load("en_core_web_sm")
from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask import Flask, render_template, request, jsonify
app = Flask(__name__, template_folder="templates",static_folder='static')

# Load fine-tuned DistilBERT for question answering (Replace with your model path)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")


# Load diseases data from JSON file
def load_responses(file_path="Untitled-1.json"):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not isinstance(data, dict) or 'diseases' not in data:
                raise ValueError("Invalid JSON structure. Expected a 'diseases' key.")
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON. Check the structure of '{file_path}'.")
        return {}
    except ValueError as e:
        print(f"Error: {e}")
        return {}

# Extract symptoms using trained NER model
def extract_symptoms(user_input):
    # Use spaCy NER to extract symptoms
    doc = nlp(user_input)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
    print(f"Extracted symptoms from NER: {symptoms}")  # Debug: Log extracted symptoms

    # Fallback: Treat the input as comma-separated symptoms
    if not symptoms:
        symptoms = [sym.strip() for sym in user_input.split(",") if sym.strip()]
        print(f"Using fallback mechanism. Symptoms: {symptoms}")  # Debug: Log fallback

    return symptoms



import spacy
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags
from spacy.tokens import DocBin
from spacy.tokens import Doc

def create_spacy_file(disease_list, output_path):
    nlp = spacy.blank("en")
    disease_list = [
    {
        "name": "Flu",
        "type": "disease",
        "symptoms": ["fever", "cough"],
        "causes": "virus",
        "treatment": "rest and fluids",
    }
     ]

    output_path = r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output\Untitled-1.spacy"
    doc_bin = DocBin()
    for disease in disease_list:
        context = f"{disease['name']} is a {disease['type']}. Symptoms: {', '.join(disease['symptoms'])}. Causes: {disease['causes']}. Treatment: {disease['treatment']}."
        doc = nlp.make_doc(context)
        annotations = {"entities": []}

        # Match symptoms and calculate offsets
        for symptom in disease["symptoms"]:
            start = context.find(symptom)
            if start != -1:
                end = start + len(symptom)
                annotations["entities"].append((start, end, "SYMPTOM"))

        # Only add examples with valid entities
        if annotations["entities"]:
            try:
                example = Example.from_dict(doc, annotations)
                doc_bin.add(example)
            except ValueError as e:
                print(f"Error creating example for context: {context}")
                print(f"Annotations: {annotations}")
                print(f"Error: {e}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc_bin.to_disk(r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output")
    print(f"Training data saved to {output_path}")


import spacy
from spacy.tokens import DocBin

def validate_spacy_file(file_path):
    nlp = spacy.blank("en")  # Blank English model
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    print(f"Loaded {len(docs)} documents from {file_path}.")
    for doc in docs[:5]:  # Display the first 5 docs
        print(doc.text, [(ent.text, ent.label_) for ent in doc.ents])

# Use a raw string for the file path to avoid escape sequence issues
validate_spacy_file(r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output\Untitled-1.spacy")



def train_ner_model():
    # Define your training data
    training_data = [
    ("The patient has a fever and sore throat.", {"entities": [(17, 22, "SYMPTOM"), (27, 38, "SYMPTOM")]}),
    ("I am experiencing nausea and vomiting.", {"entities": [(18, 24, "SYMPTOM"), (29, 37, "SYMPTOM")]}),
    ("She has persistent fatigue and muscle ache.", {"entities": [(9, 26, "SYMPTOM"), (31, 43, "SYMPTOM")]}),
    ("He complained about joint pain and stiffness.", {"entities": [(20, 30, "SYMPTOM"), (35, 44, "SYMPTOM")]}),
    ("Symptoms include chest pain and shortness of breath.", {"entities": [(17, 27, "SYMPTOM"), (32, 51, "SYMPTOM")]}),
    ("I have been feeling dizzy and lightheaded.", {"entities": [(14, 18, "SYMPTOM"), (23, 35, "SYMPTOM")]}),
    ("The cough has been persistent and dry.", {"entities": [(4, 8, "SYMPTOM"), (23, 26, "SYMPTOM")]}),
    ("He has a runny nose and sneezing.", {"entities": [(10, 18, "SYMPTOM"), (23, 30, "SYMPTOM")]}),
    ("She is suffering from abdominal pain and bloating.", {"entities": [(21, 28, "SYMPTOM"), (33, 40, "SYMPTOM")]}),
    ("I feel nauseous and have lost my appetite.", {"entities": [(7, 14, "SYMPTOM"), (19, 34, "SYMPTOM")]}),
    ("The patient reports severe headaches and blurred vision.", {"entities": [(21, 31, "SYMPTOM"), (36, 51, "SYMPTOM")]}),
    ("He is complaining of a sore throat and earache.", {"entities": [(19, 28, "SYMPTOM"), (33, 40, "SYMPTOM")]}),
    ("The flu symptoms include chills and muscle aches.", {"entities": [(16, 21, "SYMPTOM"), (26, 37, "SYMPTOM")]}),
    ("She is feeling weak and has no energy.", {"entities": [(14, 18, "SYMPTOM"), (23, 30, "SYMPTOM")]}),
    ("The infection causes a rash and swollen lymph nodes.", {"entities": [(20, 24, "SYMPTOM"), (29, 50, "SYMPTOM")]}),
    ("I have difficulty breathing and a tight chest.", {"entities": [(12, 20, "SYMPTOM"), (25, 35, "SYMPTOM")]}),
    ("She has severe back pain and leg numbness.", {"entities": [(9, 18, "SYMPTOM"), (23, 34, "SYMPTOM")]}),
    ("The patient is experiencing shortness of breath and fatigue.", {"entities": [(28, 49, "SYMPTOM"), (15, 22, "SYMPTOM")]}),
    ("I have a headache and feel nauseous.", {"entities": [(5, 13, "SYMPTOM"), (18, 25, "SYMPTOM")]}),
    ("The child has stomach cramps and diarrhea.", {"entities": [(16, 24, "SYMPTOM"), (29, 38, "SYMPTOM")]}),
    ("He is dealing with sore muscles and joint stiffness.", {"entities": [(23, 30, "SYMPTOM"), (35, 50, "SYMPTOM")]}),
    ("The patient has a cold with a sore throat and congestion.", {"entities": [(17, 22, "SYMPTOM"), (27, 38, "SYMPTOM"), (43, 53, "SYMPTOM")]}),
    ("She reports a persistent cough and fatigue.", {"entities": [(16, 20, "SYMPTOM"), (25, 32, "SYMPTOM")]}),
    ("I feel lightheaded and my heart is racing.", {"entities": [(7, 18, "SYMPTOM"), (23, 38, "SYMPTOM")]}),
    ("The patient has difficulty swallowing and a hoarse voice.", {"entities": [(28, 42, "SYMPTOM"), (47, 59, "SYMPTOM")]}),
    ("He is experiencing abdominal cramps and bloating.", {"entities": [(21, 26, "SYMPTOM"), (31, 38, "SYMPTOM")]}),
    ("She has persistent ringing in the ears and dizziness.", {"entities": [(9, 35, "SYMPTOM"), (40, 48, "SYMPTOM")]}),
    ("The symptoms include nausea and stomach pain.", {"entities": [(18, 24, "SYMPTOM"), (29, 41, "SYMPTOM")]}),
    ("He is feeling weak and has a headache.", {"entities": [(14, 18, "SYMPTOM"), (23, 31, "SYMPTOM")]}),
    ("The patient reports body aches and chills.", {"entities": [(21, 31, "SYMPTOM"), (36, 41, "SYMPTOM")]}),
    ("She has a runny nose and mild cough.", {"entities": [(10, 18, "SYMPTOM"), (23, 27, "SYMPTOM")]}),
    ("I feel fatigued and have muscle soreness.", {"entities": [(7, 15, "SYMPTOM"), (20, 37, "SYMPTOM")]}),
    ("The fever is high, and she feels weak.", {"entities": [(4, 9, "SYMPTOM"), (14, 18, "SYMPTOM")]}),
    ("He complains of dizziness and nausea.", {"entities": [(17, 25, "SYMPTOM"), (30, 37, "SYMPTOM")]}),
    ("The patient has a sore throat and swollen glands.", {"entities": [(17, 22, "SYMPTOM"), (27, 43, "SYMPTOM")]}),
    ("She has back pain and difficulty moving.", {"entities": [(9, 17, "SYMPTOM"), (22, 39, "SYMPTOM")]}),
    ("The infection caused a headache and fever.", {"entities": [(20, 28, "SYMPTOM"), (33, 38, "SYMPTOM")]}),
    ("He has severe nausea and vomiting.", {"entities": [(9, 15, "SYMPTOM"), (20, 27, "SYMPTOM")]}),
    ("The patient has a rash and fever.", {"entities": [(17, 21, "SYMPTOM"), (26, 31, "SYMPTOM")]}),
    ("She is experiencing pain in her joints and muscles.", {"entities": [(20, 30, "SYMPTOM"), (35, 44, "SYMPTOM")]}),
    ("I feel nauseous and dizzy.", {"entities": [(7, 14, "SYMPTOM"), (19, 24, "SYMPTOM")]}),
    ("The patient has stomach cramps and indigestion.", {"entities": [(17, 25, "SYMPTOM"), (30, 41, "SYMPTOM")]}),
    ("He reports chest pain and difficulty breathing.", {"entities": [(17, 26, "SYMPTOM"), (31, 48, "SYMPTOM")]}),
    ("She has a dry cough and a sore throat.", {"entities": [(9, 14, "SYMPTOM"), (19, 30, "SYMPTOM")]}),
    ("The patient feels weak and fatigued.", {"entities": [(16, 20, "SYMPTOM"), (25, 34, "SYMPTOM")]}),
    ("I have joint pain and muscle stiffness.", {"entities": [(7, 16, "SYMPTOM"), (21, 39, "SYMPTOM")]}),
    ("The flu symptoms include chills, fever, and body aches.", {"entities": [(16, 21, "SYMPTOM"), (26, 31, "SYMPTOM"), (36, 47, "SYMPTOM")]}),
    ("She has abdominal discomfort and bloating.", {"entities": [(9, 28, "SYMPTOM"), (33, 40, "SYMPTOM")]}),
    ("He is suffering from headaches and muscle tension.", {"entities": [(9, 18, "SYMPTOM"), (23, 38, "SYMPTOM")]}),
    ("The patient has throat irritation and a cough.", {"entities": [(17, 34, "SYMPTOM"), (39, 43, "SYMPTOM")]}),
    ("She complains of nausea and back pain.", {"entities": [(18, 24, "SYMPTOM"), (29, 38, "SYMPTOM")]}),
    ("I feel congested and have a sore throat.", {"entities": [(7, 16, "SYMPTOM"), (21, 30, "SYMPTOM")]}),
    ("He has severe fatigue and joint pain.", {"entities": [(9, 18, "SYMPTOM"), (23, 33, "SYMPTOM")]}),
    ("The patient has muscle aches and chills.", {"entities": [(17, 28, "SYMPTOM"), (33, 38, "SYMPTOM")]}),
    ("She is suffering from chronic pain and insomnia.", {"entities": [(21, 26, "SYMPTOM"), (31, 40, "SYMPTOM")]}),
    ("I have a headache and feel dizzy.", {"entities": [(5, 13, "SYMPTOM"), (18, 23, "SYMPTOM")]}),
    ("The child is complaining of stomach pain and nausea.", {"entities": [(25, 36, "SYMPTOM"), (41, 47, "SYMPTOM")]}),
]


    # Step 1: Create a blank spaCy model
    nlp = spacy.blank("en")  # Blank English pipeline
    ner = nlp.add_pipe("ner")  # Add NER pipeline component
    ner.add_label("SYMPTOM")
    # Step 2: Add labels to the NER component
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])
    doc_bin = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)  
        example = Example.from_dict(doc, annotations)  
        doc_bin.add(example.reference)  
    output_path = (r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output\Untitled-1.spacy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    doc_bin.to_disk(output_path)
    print(f"Training data saved to: {output_path}")
    config_path = (r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\base_config.cfg")
    output_dir = (r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output")

    try:
        subprocess.run(
            ["python", "-m", "spacy", "train", config_path, "--output", output_dir],
            check=True
        )
        print(f"NER model training complete. Model saved in '{output_dir}/model-best'")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")

def validate_training_data(training_data):
    nlp = spacy.blank("en")  # Blank English pipeline
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        entities = annotations["entities"]
        try:
            tags = offsets_to_biluo_tags(doc, entities)
            print(f"Valid: {text} -> {tags}")
        except Exception as e:
            print(f"Invalid: {text} -> {entities}")
            print(f"Error: {e}")
# Match diseases using fuzzy matching
def match_disease_fuzzy(symptoms, diseases, threshold=1, score_cutoff=60, max_results=5):
    if not diseases:
        return "No disease data is available. Please check the system configuration."
    if not symptoms:
        return "No symptoms provided. Please describe your symptoms."

    print(f"Symptoms provided: {symptoms}")  # Debug log

    matched_diseases = []
    for disease in diseases:
        disease_symptoms = [symptom.lower() for symptom in disease['symptoms']]
        match_count = sum([
            1 for symptom in symptoms
            if process.extractOne(symptom.lower(), disease_symptoms, score_cutoff=score_cutoff)
        ])

        print(f"Matching disease: {disease['name']} | Match count: {match_count}")  # Debug log

        if match_count >= threshold:
            matched_diseases.append((disease['name'], disease['symptoms'], match_count))

    print(f"Matched diseases: {matched_diseases}")  # Debug log

    if not matched_diseases:
        return "I couldn't match any diseases to those symptoms. Please check if the symptoms are correct."

    matched_diseases.sort(key=lambda x: x[2], reverse=True)
    response = "Based on the symptoms you provided, the most likely disease(s) could be:\n"
    for disease, symptoms, match_count in matched_diseases[:max_results]:
        symptoms_list = ", ".join(symptoms)
        response += f"- **{disease}**\n   Matched Symptoms: {match_count}\n   Key Symptoms: {symptoms_list}\n\n"

    return response



def answer_question(user_input, diseases):
    context = " ".join(
        f"{disease['name']} is a {disease['type']}. Symptoms: {', '.join(disease['symptoms'])}. "
        f"Causes: {disease['causes']}. Treatment: {disease['treatment']}. Prevention: {disease['prevention']}."
        for disease in diseases
    )

    result = qa_pipeline(question=user_input, context=context)
    return result['answer'] if result else "I'm sorry, I couldn't find an answer to your question."

def fine_tune_qa_model():
    from datasets import Dataset
    from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments
    import torch

    # Load disease data from the JSON file
    data = load_responses("Untitled-1.json").get("diseases", [])
    if not data:
        print("Error: No disease data found. Ensure the JSON file contains valid data.")
        return

    # Prepare contexts, questions, and answers
    contexts, questions, answers = [], [], []
    for disease in data:
        context = f"{disease['name']} is a {disease['type']}. Symptoms: {', '.join(disease['symptoms'])}. " \
                  f"Causes: {disease['causes']}. Treatment: {disease['treatment']}."
        symptoms_str = ", ".join(disease["symptoms"])
        start_idx = context.find(symptoms_str)
        
        if start_idx == -1:
            print(f"Warning: Symptoms '{symptoms_str}' not found in context: '{context}'")
            continue  # Skip if symptoms cannot be located in context
        
        end_idx = start_idx + len(symptoms_str)
        contexts.append(context)
        questions.append(f"What are the symptoms of {disease['name']}?")
        answers.append({"text": symptoms_str, "start": start_idx, "end": end_idx})

    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    def preprocess(example):
        tokenized = tokenizer(
            example["question"],
            example["context"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        start = example["answers"]["start"]
        end = example["answers"]["end"]
        tokenized["start_positions"] = tokenized.char_to_token(0, start)
        tokenized["end_positions"] = tokenized.char_to_token(0, end)

        if tokenized["start_positions"] is None or tokenized["end_positions"] is None:
            print(f"Warning: Token mapping failed for question: {example['question']} and context: {example['context']}")
            tokenized["start_positions"], tokenized["end_positions"] = 0, 0  # Assign default positions to avoid crashing
        
        return tokenized

    dataset = Dataset.from_dict({"context": contexts, "question": questions, "answers": answers})
    dataset = dataset.map(preprocess)

    # Load the pre-trained model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qa_model",
        eval_strategy="epoch",  # Fixed deprecation warning
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=lambda features: {
            k: torch.tensor([f[k] for f in features if f[k] is not None]) for k in features[0]
        },
    )

    # Train the model
    print("Starting model fine-tuning...")
    try:
        trainer.train()
        print("Model fine-tuning completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save the trained model
    model.save_pretrained("./fine_tuned_qa_model")
    tokenizer.save_pretrained("./fine_tuned_qa_model")
    print("QA model fine-tuning complete. Model saved in './fine_tuned_qa_model'")



def validate_entities(disease_list):
    nlp = spacy.blank("en")
    for disease in disease_list:
        context = f"{disease['name']} is a {disease['type']}. Symptoms: {', '.join(disease['symptoms'])}. Causes: {disease['causes']}. Treatment: {disease['treatment']}."
        doc = nlp.make_doc(context)
        entities = []
        for symptom in disease["symptoms"]:
            start = context.find(symptom)
            if start != -1:
                end = start + len(symptom)
                entities.append((start, end, "SYMPTOM"))
        try:
            tags = offsets_to_biluo_tags(doc, entities)
            print(f"Valid: {tags}")
        except Exception as e:
            print(f"Error with entities: {entities}")
            print(f"Context: {context}")
            print(f"Error: {e}")

    validate_entities([
        {
            "name": "Flu",
            "type": "disease",
            "symptoms": ["fever", "cough"],
            "causes": "virus",
            "treatment": "rest and fluids",
        }
    ])
@app.route('/chat', methods=['POST'])
def chatbot():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'type': 'error', 'message': 'Invalid request'}), 400

    user_message = data['message']
    # Process the message here and generate a response
    bot_response = f"You said: {user_message}"
    return jsonify({'type': 'qa', 'message': bot_response})
# Main chatbot function
def main():
    print("Welcome to your NLP-based medical chatbot. Type 'quit' to exit.")
    file_path = "Untitled-1.json"
    data = load_responses(file_path)
    app.run(debug=True)
    if not data:
        print("Unable to load disease data. Exiting.")
        return

    diseases = data['diseases']
    
    while True:
        print("\nYou can ask me about diseases, provide symptoms, or train the models.")
        user_input = input("You: ").strip().lower()

        # Initialize the response variable to handle cases where no condition is met
        response = "I'm sorry, I didn't understand that."

        if user_input == "quit":
            print("Goodbye! Take care.")
            break

        elif user_input == "train the models":
            print("Training models... Please wait.")
            train_ner_model()  # Call the NER training function
            fine_tune_qa_model()  # Call the QA fine-tuning function
            response = "Models trained and saved successfully."

        elif user_input == "train_ner":
            print("Training NER model...")
            train_ner_model()  # Call the NER training function
            response = "NER model trained and saved successfully."

        elif user_input == "train_qa":
            print("Training QA model...")
            fine_tune_qa_model()  # Call the QA fine-tuning function
            response = "QA model trained and saved successfully."

        elif "symptom" in user_input:
            symptoms_input = input("Please describe your symptoms: ")
            extracted_symptoms = extract_symptoms(symptoms_input)
            if extracted_symptoms:
                response = match_disease_fuzzy(extracted_symptoms, diseases)
            else:
                response = "I couldn't detect any symptoms in your input. Please try again."
        else:
            response = answer_question(user_input, diseases)

        print("\nChatbot:", response)

from flask import Flask, send_from_directory    
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
@app.route('/')
def home():
    return render_template('index.html')

diseases = load_responses("Untitled-1.json").get("diseases", [])

@app.route('/chat', methods=['POST'])
def chat():
    # Load user message
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'reply': 'Please provide a message.'}), 400

    # Ensure diseases data is loaded
    if not diseases:
        return jsonify({'reply': 'Disease data is unavailable. Please check the system configuration.'})

    # Check if the message is about a specific disease
    name = next((disease['name'] for disease in diseases if disease['name'].lower() in user_message.lower()), None)
    if name:
        disease = next(d for d in diseases if d['name'] == name)
        reply = (
            f"**{disease['name']}**\n"
            f"- **Type:** {disease['type']}\n"
            f"- **Symptoms:** {', '.join(disease['symptoms'])}\n"
            f"- **Causes:** {disease['causes']}\n"
            f"- **Treatment:** {disease['treatment']}\n"
            f"- **Cure:** {disease.get('cure', 'Not specified')}\n"
            f"- **Prevention:** {disease.get('prevention', 'Not specified')}"
        )
        return jsonify({'reply': reply})

    # Process symptom-based queries
    elif "symptom" in user_message.lower():
        extracted_symptoms = extract_symptoms(user_message)
        if extracted_symptoms:
            bot_reply = match_disease_fuzzy(extracted_symptoms, diseases)
        else:
            bot_reply = "I couldn't detect any symptoms. Please provide more details."

    # Fallback for general queries
    else:
        bot_reply = (
            "I'm sorry, I couldn't understand your request. You can ask me about a disease "
            "or describe your symptoms for assistance."
        )
    
    return jsonify({'reply': bot_reply})


@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
if __name__ == '__main__':
    app.run(debug=True)
    validate_spacy_file(r"C:\Users\vaibhav gupta\OneDrive\Desktop\All Projects\project exhibition\output\Untitled-1.spacy")
    



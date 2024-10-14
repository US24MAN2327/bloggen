from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, db
import uuid
from datetime import datetime
import logging

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("blogssave-firebase-adminsdk-3qp96-032e632e33.json")  # Replace with your JSON key path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://blogssave-default-rtdb.firebaseio.com/'  # Replace with your database URL
})

# Load the model and tokenizer from Hugging Face locally
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


# Function to generate text locally using the loaded model
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")  # Tokenize the input
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=300,  # You can adjust the maximum length
        temperature=0.7,  # Adjust the temperature for creativity
        do_sample=True,  # Enable sampling for more diverse results
        top_k=50,  # Consider only the top 50 tokens at each step
        top_p=0.95,  # Nucleus sampling
    )
    
    # Decode the generated sequence back into text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

# Function to save blog to Firebase
def save_blog_to_firebase(title, content):
    try:
        blog_id = str(uuid.uuid4())  # Generate a unique ID for each blog
        blog_data = {
            "id": blog_id,
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat()  # Save the current timestamp
        }

        # Save to Firebase Realtime Database
        ref = db.reference(f'/blogs/{blog_id}')
        ref.set(blog_data)  # Save the blog data
        logging.info(f"Blog saved to Firebase with ID: {blog_id}")  # Log the successful save
    except Exception as e:
        logging.error(f"Error saving to Firebase: {e}")  # Log any errors during the save process

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_blog', methods=['POST'])
def generate_blog():
    prompt = request.form.get('prompt')
    title = prompt  # You can use the prompt as the title or modify it as needed
    generated_text = generate_text(prompt)
    
    # Save the generated blog post with title to Firebase
    save_blog_to_firebase(title, generated_text)
    
    return jsonify({'blog_content': generated_text})

# Route to render the Firebase Dashboard
@app.route('/dashboard')
def dashboard():
    try:
        # Fetch all blog posts from Firebase
        ref = db.reference('/blogs')  # Reference to the blogs node
        blogs_data = ref.get()  # Get all the blogs stored
        
        if blogs_data is None:
            blogs_data = {}  # If no data is found, use an empty dictionary
            
        logging.info("Fetched blog data from Firebase")  # Log success
        
        # Pass the blogs data to the template
        return render_template('firebase_dashboard.html', blogs=blogs_data)
    
    except Exception as e:
        logging.error(f"Error fetching blog data from Firebase: {e}")
        return jsonify({'error': 'Error fetching blog data'}), 500


if __name__ == '__main__':
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)

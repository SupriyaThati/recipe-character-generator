# Created by Supriya Thati, 2025
# Do not distribute without attribution

import json
import os
import re
import base64
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client for OpenRouter (for text generation)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it in a .env file or your environment.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Load Hugging Face API key (for image generation)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable is not set. Please set it in a .env file or your environment.")

# --- FINAL FIXED CODE ---
# Use a model that is available on the free public Inference API.
# stabilityai/stable-diffusion-2-1 is a good choice.
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"


# Function to sanitize prompts
def sanitize_prompt(prompt):
    """Remove excessive whitespace and limit prompt length."""
    prompt = " ".join(prompt.split())
    return prompt[:500]

# Retry decorator for Hugging Face API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
def make_huggingface_request(prompt):
    response = requests.post(
        HUGGINGFACE_API_URL,
        headers={
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": prompt,
            "parameters": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "guidance_scale": 3.5,
                "max_sequence_length": 512
            }
        },
        timeout=60
    )
    return response

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        cuisine = data.get('cuisine', '').strip()
        ingredients = data.get('ingredients', '').strip()
        capacity = data.get('capacity', '')

        if not cuisine or not ingredients or not capacity:
            return jsonify({"error": "Cuisine, ingredients, and serving capacity are required"}), 400

        try:
            capacity = int(capacity)
            if capacity <= 0:
                raise ValueError
        except ValueError:
            return jsonify({"error": "Serving capacity must be a positive integer"}), 400

        prompt = (
            f"Create a detailed {cuisine} recipe for {capacity} servings using these ingredients: {ingredients}. "
            f"Return a JSON object with the following fields: "
            f"'name' (string, recipe name), "
            f"'ingredients' (list of strings, each with quantity and ingredient), "
            f"'instructions' (list of strings, each a clear step). "
            f"Ensure the response is valid JSON, contains only the JSON object, and is not wrapped in markdown or code fences."
        )

        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct",
            messages=[
                {"role": "system", "content": "You are a culinary expert. Always return valid JSON with the exact structure requested, without markdown or extra text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        recipe_text = response.choices[0].message.content.strip()
        if recipe_text.startswith("```json"):
            recipe_text = recipe_text[7:-3].strip()
        elif recipe_text.startswith("```"):
            recipe_text = recipe_text[3:-3].strip()

        recipe_text = re.sub(r',\s*([\]}])', r'\1', recipe_text)
        print(f"Raw recipe response: {recipe_text}")

        try:
            recipe_data = json.loads(recipe_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, Raw response: {recipe_text}")
            return jsonify({
                "name": "Unknown Recipe",
                "ingredients": ["Failed to parse recipe data"],
                "instructions": [f"Error: {str(e)}"],
                "image": ""
            }), 200

        name = recipe_data.get("name", "Unknown Recipe")
        ingredients_list = recipe_data.get("ingredients", [])
        instructions = recipe_data.get("instructions", [])

        if not isinstance(ingredients_list, list):
            ingredients_list = [str(ingredients_list)] if ingredients_list else []
        if not isinstance(instructions, list):
            instructions = [str(instructions)] if instructions else []

        ingredients_list = [str(item) for item in ingredients_list if item]
        instructions = [str(item) for item in instructions if item]
        name = str(name) if name else "Unknown Recipe"

        return jsonify({
            "name": name,
            "ingredients": ingredients_list,
            "instructions": instructions,
            "image": ""
        })

    except Exception as e:
        print(f"Error generating recipe: {e}")
        return jsonify({"error": f"Error generating recipe: {str(e)}"}), 500

@app.route('/generate_character', methods=['POST'])
def generate_character():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()

        if not description:
            return jsonify({"error": "Character description is required"}), 400

        prompt = (
            f"Generate a detailed character based on this description: {description}. "
            f"Return a JSON object with exactly these fields: "
            f"'name' (string, character's name), "
            f"'story' (list of strings, each a paragraph or key point of the character's background). "
            f"Ensure the response is valid JSON, contains only the JSON object, and is not wrapped in markdown or code fences."
        )

        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct",
            messages=[
                {"role": "system", "content": "You are a creative character generator. Always return valid JSON with the exact structure requested, without markdown or extra text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        character_text = response.choices[0].message.content.strip()
        if character_text.startswith("```json"):
            character_text = character_text[7:-3].strip()
        elif character_text.startswith("```"):
            character_text = character_text[3:-3].strip()

        character_text = re.sub(r',\s*([\]}])', r'\1', character_text)
        print(f"Raw character response: {character_text}")

        try:
            character_data = json.loads(character_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, Raw response: {character_text}")
            return jsonify({
                "name": "Unknown Character",
                "story": [f"Failed to parse character data. Error: {str(e)}"],
                "image": ""
            }), 200

        name = character_data.get("name", "Unknown Character")
        story = character_data.get("story", [])
        if not isinstance(story, list):
            story = [str(story)] if story else []
        story = [str(item) for item in story if item]
        name = str(name) if name else "Unknown Character"

        return jsonify({
            "name": name,
            "story": story,
            "image": ""
        })

    except Exception as e:
        print(f"Error generating character: {e}")
        return jsonify({"error": f"Error generating character: {str(e)}"}), 500

@app.route('/generate_recipe_image', methods=['POST'])
def generate_recipe_image():
    try:
        data = request.get_json()
        cuisine = data.get('cuisine', '').strip()
        ingredients = data.get('ingredients', '').strip()

        if not cuisine or not ingredients:
            return jsonify({"error": "Cuisine and ingredients are required for image generation"}), 400

        prompt = sanitize_prompt(
            f"A beautifully plated {cuisine} dish made with {ingredients}, vibrant colors, appetizing presentation, high detail, photo-realistic style"
        )

        response = make_huggingface_request(prompt)

        print(f"Hugging Face API response status: {response.status_code}")
        print(f"Hugging Face API response headers: {response.headers}")

        if response.status_code != 200:
            error_message = response.text or "Unknown error from Hugging Face API"
            return jsonify({"error": f"Hugging Face API error: {error_message}"}), 500

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "An error occurred during image generation.")
            except json.JSONDecodeError:
                error_message = response.text

            return jsonify({"error": f"Unexpected response from Hugging Face API: {error_message}"}), 500

        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return jsonify({"image": image_base64})

    except Exception as e:
        print(f"Error generating recipe image: {e}")
        return jsonify({"error": f"Error generating image: {str(e)}"}), 500

@app.route('/generate_character_image', methods=['POST'])
def generate_character_image():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()

        if not description:
            return jsonify({"error": "Character description is required for image generation"}), 400

        prompt = sanitize_prompt(
            f"A detailed illustration of {description}, vibrant colors, high detail, digital art style"
        )

        response = make_huggingface_request(prompt)

        print(f"Hugging Face API response status: {response.status_code}")
        print(f"Hugging Face API response headers: {response.headers}")

        if response.status_code != 200:
            error_message = response.text or "Unknown error from Hugging Face API"
            return jsonify({"error": f"Hugging Face API error: {error_message}"}), 500

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "An error occurred during image generation.")
            except json.JSONDecodeError:
                error_message = response.text
                
            return jsonify({"error": f"Unexpected response from Hugging Face API: {error_message}"}), 500

        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return jsonify({"image": image_base64})

    except Exception as e:
        print(f"Error generating character image: {e}")
        return jsonify({"error": f"Error generating image: {str(e)}"}), 500

# Health check for Render
@app.route('/healthz')
def health_check():
    return 'ok', 200

if __name__ == '__main__':
    # Use a dynamic port for production environments like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
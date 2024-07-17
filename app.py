from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import io

app = Flask(__name__)

   # Load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.route('/predict', methods=['POST'])
def predict():
       data = request.json
       image_url = data['image_url']
       question = data['question']

       # Prepare image and question
       image = Image.open(requests.get(image_url, stream=True).raw)

       # Process inputs
       encoding = processor(image, question, return_tensors="pt")

       # Forward pass
       outputs = model(**encoding)
       logits = outputs.logits
       idx = logits.argmax(-1).item()
       answer = model.config.id2label[idx]

       return jsonify({"answer": answer})

if __name__ == '__main__':
       app.run(host='0.0.0.0', port=10000)
   
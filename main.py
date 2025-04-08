import os
import csv
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def extract_classification(text):
    text = text.lower()
    if "unstaged" in text:
        return "unstaged"
    elif "staged" in text:
        return "staged"
    elif "indeterminate" in text:
        return "indeterminate"
    else:
        return "unknown"

IMAGE_FOLDER = "images"
OUTPUT_CSV = "results.csv"

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Classification"])

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            filepath = os.path.join(IMAGE_FOLDER, filename)
            print(f"Classifying: {filename}")

            with open(filepath, "rb") as img_file:
                base64_img = base64.b64encode(img_file.read()).decode("utf-8")

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional real estate photographer and virtual stager. Your task is to classify real estate photos as either 'staged', 'unstaged', or 'indeterminate'. Respond with one word only."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Classify this real estate photo:"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                            ]
                        }
                    ],
                    max_tokens=50
                )
                raw_result = response.choices[0].message.content.strip()
                cleaned_result = extract_classification(raw_result)
                print(f"→ Result: {cleaned_result}")
                writer.writerow([filename, cleaned_result])
                time.sleep(0.6)  # find the limit
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([filename, "error"])
                time.sleep(1)

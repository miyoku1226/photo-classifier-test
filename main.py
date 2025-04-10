import os
import csv
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception)
)
def classify_image(base64_img):
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
    return response.choices[0].message.content.strip()

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
                result = classify_image(base64_img)
                cleaned_result = extract_classification(result)
                print(f"→ Result: {cleaned_result}")
                writer.writerow([filename, cleaned_result])
            except Exception as e:
                print(f"Failed after retries: {e}")
                writer.writerow([filename, "error"])

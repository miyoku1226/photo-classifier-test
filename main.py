import os
import csv
import base64
from openai import OpenAI
from dotenv import load_dotenv
from app.retry import retry_with_backoff  #you can use the retry.py in app/

load_dotenv()
client = OpenAI()

@retry_with_backoff(retries=5, exceptions=(Exception,))
def classify_image(base64_img):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":(
                    "You are a professional real estate photographer and virtual stager. Your task is to classify real estate photos as either 'staged', 'unstaged', or 'indeterminate'.\n\n"

                "Classification Rules:\n"
                "1. UNSTAGED if:\n"
                "   - The room is clearly a living space (living room, bedroom, dining room, family room, sunroom) AND it is completely empty\n"
                "   - The room has no furniture, or only minimal furniture (1–2 pieces in a space meant for more)\n"
                "   - There are no personal items, no decor, and the space appears vacant or uninhabited\n"
                "   - Remodels or clean finishes (e.g., fresh paint, ceiling fans, new flooring) DO NOT make a room 'staged' unless furniture or decor is present\n\n"

                "2. STAGED if:\n"
                "   - The space contains a full furniture set appropriate for the room type (e.g., a bed and nightstands in a bedroom, sofa and coffee table in a living room)\n"
                "   - The furniture is arranged with clear intent to show how the room is used\n"
                "   - There are decorative or lifestyle elements (e.g., pillows, lamps, wall art, plants, rugs, etc.) that enhance the space visually\n"
                "   - The room looks either lived-in or professionally styled\n\n"

                "3. INDETERMINATE only if:\n"
                "   - The photo is not of a living space (e.g., kitchen, bathroom, garage, laundry room, closet, hallway, exterior)\n"
                "   - Or, the photo is too ambiguous to determine if furniture is present (e.g., blurry, dark, incomplete room view)\n\n"

                "Important Notes:\n"
                "- Do NOT mark rooms as 'indeterminate' just because they are clean, modern, or remodeled—if they are empty, mark them as 'unstaged'\n"
                "- Prioritize accurate classification over caution. Only choose 'indeterminate' if you're genuinely unsure due to the image content type\n\n"

                "Respond with EXACTLY one word: 'staged', 'unstaged', or 'indeterminate'"
            )
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

def run_classification(image_folder="images", output_csv="results.csv"):
    existing = set()

    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    existing.add(row[0])

    with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
            writer.writerow(["Filename", "Classification"])

        for filename in os.listdir(image_folder):
            if filename in existing:
                continue
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                filepath = os.path.join(image_folder, filename)
                print(f"Classifying: {filename}")

                with open(filepath, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode("utf-8")

                try:
                    result = classify_image(base64_img)
                    cleaned = extract_classification(result)
                    print(f"→ Result: {cleaned}")
                    writer.writerow([filename, cleaned])
                except Exception as e:
                    print(f"Failed after retries: {e}")
                    writer.writerow([filename, "error"])

if __name__ == "__main__":
    run_classification()

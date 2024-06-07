"""
I was expecting this to be harder...

This will just dump all of the text in the image to a single string. 
It will not be context aware. GPT-4 is really expensive and probably doesn't work 
for this kind of use case. 
I don't think there is a good enough OSS multi-modal model though that could work instead.
"""

import pytesseract
from PIL import Image


def extract_text(image_path):
    # Load the image from the given path
    image = Image.open(image_path)

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)

    return text


# Example usage
if __name__ == "__main__":
    path = input("Enter the path to the image: ")
    extracted_text = extract_text(path)
    print("Extracted Text:")
    print(extracted_text)

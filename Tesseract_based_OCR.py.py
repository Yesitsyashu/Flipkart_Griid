import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np

# Set the path to the Tesseract executable (adjust the path to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
img_path = r'E:\Flipkart\ocr\check_img_head&shoulder.webp'
img = cv2.imread(img_path)

# Step 1: Resize the image (increase size by 2x for small text)
img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Step 3: Apply sharpening to enhance the edges of the text
kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])
sharpened = cv2.filter2D(gray, -1, kernel)

# Step 4: Apply denoising to clean the image (if necessary)
denoised = cv2.fastNlMeansDenoising(sharpened, None, 30, 7, 21)

# Step 5: Apply simple thresholding
_, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)

# Optional: Display the processed image
cv2.imshow('Processed Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 6: Extract text using Tesseract (set language if needed)
text = pytesseract.image_to_string(thresh, lang='eng')

# Print the extracted text
print("Extracted Text: ", text)

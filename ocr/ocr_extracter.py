import pytesseract
from PIL import Image
import re
import os



def preprocess(image):
    #--- Light preprocessing ---
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((image.width * 2, image.height * 2))  # Slight upscale
    image = image.point(lambda x: 0 if x < 140 else 255, '1')  # Light binarization
    return image


def extract_text(image_path):
    """Extract text from an image using OCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f" Image not found: {image_path}")
    image = Image.open(image_path)

    img_p=preprocess(image)
    
    # Run OCR
    text = pytesseract.image_to_string(img_p, lang="eng")

    
    #print(text)
    return text.strip()


def clean_ocr_text(text):
    """Clean OCR text (remove weird chars and normalize spacing)."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    #print(text)
    return text.strip()




def extract_key_info(text):
    """Robust key information extractor for various product labels."""
    summary = {}
    text_lower = text.lower()

    # --- Product Name ---
    # First non-empty line with >1 word and without keywords
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if not re.search(r'(batch|lot|mfg|exp|mrp|price|net|content|manufactured|marketed)', ln, re.I) and len(ln.split()) > 1:
            summary["Product Name"] = ln.strip()
            break

    # --- Batch / Lot ---
    batch_pattern = re.search(
        r'\b(?:batch|lot)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_/]+)',
        text, re.I)
    if batch_pattern:
        summary["Batch/Lot No."] = batch_pattern.group(1).strip()

    # --- Manufacture Date ---
    mfg_pattern = re.search(
        r'\b(?:mfg\.?|mfd\.?|manufactured|mfg\s*month)\b[^:]*[:\-]?\s*([A-Z]{3,9}[- ]?\d{2,4}|\d{1,2}[- /.]\d{4})',
        text, re.I)
    if mfg_pattern:
        summary["Manufacture Date"] = mfg_pattern.group(1).upper().strip()

    # --- Expiry Date ---
    exp_pattern = re.search(
        r'\b(?:exp\.?|expiry|expd|exp\s*month)\b[^:]*[:\-]?\s*([A-Z]{3,9}[- ]?\d{2,4}|\d{1,2}[- /.]\d{4})',
        text, re.I)
    if exp_pattern:
        summary["Expiry Date"] = exp_pattern.group(1).upper().strip()

    # --- MRP / Price ---
    price_pattern = re.search(
        r'\b(?:mrp|price|rs\.?|₹)\s*(?:rs\.?)?[:\-]?\s*([0-9]+(?:[.,][0-9]{1,2})?)',
        text, re.I)
    if price_pattern:
        summary["MRP"] = f"₹{price_pattern.group(1)}"

    # --- Net Content / Quantity ---
    net_pattern = re.search(
        r'\bnet\s*(?:content|qty|quantity|weight|wt\.?)\b[^:]*[:\-]?\s*([A-Za-z0-9 .]+)',
        text, re.I)
    if net_pattern:
        summary["Net Content"] = net_pattern.group(1).strip()

    # --- Manufacturer / Marketed By ---
    manuf_pattern = re.search(
        r'\b(?:manufactured\s*by|mfg\.?\s*by|manufacturer)\b[:\-]?\s*([A-Za-z0-9 ,.&()\-]+)',
        text, re.I)
    if manuf_pattern:
        summary["Manufactured By"] = manuf_pattern.group(1).strip()

    marketed_pattern = re.search(
        r'\b(?:marketed\s*by|distributed\s*by|sold\s*by)\b[:\-]?\s*([A-Za-z0-9 ,.&()\-]+)',
        text, re.I)
    if marketed_pattern:
        summary["Marketed By"] = marketed_pattern.group(1).strip()

    # If "Marketed by" not found, try to detect a company-like name near the end
    if "Marketed By" not in summary:
        possible_brands = re.findall(
            r'\b[A-Z][a-zA-Z]+\s+(?:Biotech|Healthcare|Pharma|Labs?|Ltd|Pvt|Care)\b',
            text)
        if possible_brands:
            summary["Marketed By"] = possible_brands[-1].strip()

    # --- Address ---
    addr_pattern = re.search(
    r'(?:plot|sector|road|street|village|block|noida|delhi|karnal|mumbai|chennai|bengaluru|hyderabad|gurgaon)[A-Za-z0-9 ,.\-/]*',
    text,
    re.I)

    if addr_pattern:
        address = addr_pattern.group(0).strip()
        # --- Clean trailing parts like "Mfg. Lic. No.", "Batch", etc.
        address = re.split(
            r'(?:mfg|lic|no\.|batch|exp|mrp|manufactured|marketed|tel|ph|email)',
            address,
            flags=re.I)[0].strip(" ,.-")

        # --- Normalize spaces and punctuation
        address = re.sub(r'\s{2,}', ' ', address)
        address = re.sub(r'\s([,.-])', r'\1', address)

        summary["Address"] = address


    return summary

def summarize_image(image_path):
    """Complete flow: OCR → Clean → Extract Info → Display."""
    print("Extracting text from image...")
    text = extract_text(image_path)
    print("\nOCR completed.\n")

    text = clean_ocr_text(text)
    info = extract_key_info(text)

    print("--- Important Product Details ---")
    if info:
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print("No key product details detected.")
    print("\n Summary generated successfully.")


if __name__ == "__main__":
    # test image path
    image_path = "test2.jpeg"
    summarize_image(image_path)

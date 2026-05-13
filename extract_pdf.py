import pdfplumber
import os

pdf_path = r'C:\Users\prana\Downloads\new_rl\A_Distributed_Model-Free_Ride-Sharing_Approach_for_Joint_Matching_Pricing_and_Dispatching_Using_Deep_Reinforcement_Learning.pdf'
txt_path = r'C:\Users\prana\Downloads\new_rl\pdf_text.txt'

try:
    with pdfplumber.open(pdf_path) as pdf:
        text = '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("PDF extraction successful.")
except Exception as e:
    print(f"PDF extraction failed: {e}")

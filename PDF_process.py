import os
import pandas as pd
import PyPDF2
from dotenv import load_dotenv
import requests
import fitz 
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from pdf2image import convert_from_path

def crop_image(element, pageObj):
    [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1] 
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)

def convert_to_images(input_file,):
    images = convert_from_path(input_file,500)
    image = images[0]
    output_file = "PDF_image.png"
    image.save(output_file, "PNG")

def image_to_text(image_path):
    # Read the image
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

data_path = "pdf_temp" 
for file in os.listdir(data_path):
    if not file.endswith("pdf"): continue
    file_name = os.path.join(data_path,file)

    pdfFileObj = open(file_name, 'rb')
    print("read file: ", file_name)
    pdfReaded = PyPDF2.PdfReader(pdfFileObj)


    doc = fitz.open(file_name)
    n = doc.page_count

    doc_content = ""
    for i in range(0, n):
        
        page_n = doc.load_page(i)
        tabs = page_n.find_tables()
        page_content = page_n.get_text("blocks")
        page_info = ""
        for element in page_content:
            
            if element[6] == 0:
                page_info += element[4]
            else:
                pageObj = pdfReaded.pages[i]
                crop_image(element, pageObj)
                convert_to_images('cropped_image.pdf')
                image_text = image_to_text('PDF_image.png')
                page_info += image_text
        doc_content += page_info + "\n"
    
    txt_file = "pdf_file_data/"+file.split("pdf")[0]+'txt'
    print("saved file: ", txt_file)
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(doc_content)







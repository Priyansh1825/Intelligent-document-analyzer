import PyPDF2
from docx import Document
import os
from typing import Union

class DocumentReader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def read_document(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._read_pdf(file_path)
        elif file_ext == '.docx':
            return self._read_docx(file_path)
        elif file_ext == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
    
    def _read_pdf(self, file_path: str) -> dict:
        text = ""
        metadata = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'pages': len(pdf_reader.pages),
                'author': pdf_reader.metadata.get('/Author', ''),
                'title': pdf_reader.metadata.get('/Title', ''),
                'subject': pdf_reader.metadata.get('/Subject', '')
            }
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
    
    def _read_docx(self, file_path: str) -> dict:
        doc = Document(file_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        metadata = {
            'pages': len(doc.sections),
            'author': doc.core_properties.author,
            'title': doc.core_properties.title,
            'subject': doc.core_properties.subject
        }
        
        return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
    
    def _read_txt(self, file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        metadata = {
            'pages': 1,
            'author': '',
            'title': os.path.basename(file_path),
            'subject': ''
        }
        
        return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
    
    def is_supported_format(self, file_path: str) -> bool:
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_formats
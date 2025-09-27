import os
from typing import Dict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import PyPDF2 # type: ignore
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available. PDF support disabled.")

try:
    from docx import Document # type: ignore
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("python-docx not available. DOCX support disabled.")

class DocumentReader:
    def __init__(self):
        self.supported_formats = []
        if PDF_AVAILABLE:
            self.supported_formats.append('.pdf')
        if DOCX_AVAILABLE:
            self.supported_formats.append('.docx')
        self.supported_formats.append('.txt')
    
    def read_document(self, file_path: str) -> Dict:
        """Read document and return text with metadata"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf' and PDF_AVAILABLE:
            return self._read_pdf(file_path)
        elif file_ext == '.docx' and DOCX_AVAILABLE:
            return self._read_docx(file_path)
        elif file_ext == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_ext}. Available support: {self.supported_formats}")
    
    def _read_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available for PDF reading")
        
        text = ""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {
                    'pages': len(pdf_reader.pages),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'subject': pdf_reader.metadata.get('/Subject', '')
                }
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
    
    def _read_docx(self, file_path: str) -> Dict:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available for DOCX reading")
        
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            
            metadata = {
                'pages': len(doc.sections) if hasattr(doc, 'sections') else 1,
                'author': getattr(doc.core_properties, 'author', ''),
                'title': getattr(doc.core_properties, 'title', ''),
                'subject': getattr(doc.core_properties, 'subject', '')
            }
            
            return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def _read_txt(self, file_path: str) -> Dict:
        """Read text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        
        metadata = {
            'pages': 1,
            'author': '',
            'title': os.path.basename(file_path),
            'subject': ''
        }
        
        return {'text': text.strip(), 'metadata': metadata, 'file_path': file_path}
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_formats
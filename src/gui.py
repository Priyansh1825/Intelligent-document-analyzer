import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from src.document_reader import DocumentReader
from src.ai_processor import AIProcessor
import threading

class DocumentAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Document Analyzer")
        self.root.geometry("1000x700")
        
        self.document_reader = DocumentReader()
        self.ai_processor = AIProcessor()
        self.current_document = None
        
        self.setup_gui()
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        file_frame = ttk.LabelFrame(main_frame, text="Document Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Browse Document", 
                  command=self.browse_document).grid(row=0, column=0, padx=(0, 10))
        
        self.file_path_label = ttk.Label(file_frame, text="No document selected")
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        file_frame.columnconfigure(1, weight=1)
        
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        ttk.Button(controls_frame, text="Analyze Document", 
                  command=self.analyze_document).grid(row=0, column=0, padx=(0, 10))
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stats_frame = ttk.Frame(self.notebook, padding="10")
        self.stats_text = scrolledtext.ScrolledText(stats_frame, width=80, height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(stats_frame, text="Basic Statistics")
        
        keywords_frame = ttk.Frame(self.notebook, padding="10")
        self.keywords_text = scrolledtext.ScrolledText(keywords_frame, width=80, height=15)
        self.keywords_text.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(keywords_frame, text="Keywords")
        
        summary_frame = ttk.Frame(self.notebook, padding="10")
        self.summary_text = scrolledtext.ScrolledText(summary_frame, width=80, height=15)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(summary_frame, text="Summary")
        
        qa_frame = ttk.Frame(self.notebook, padding="10")
        qa_frame.columnconfigure(1, weight=1)
        
        ttk.Label(qa_frame, text="Question:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.question_entry = ttk.Entry(qa_frame)
        self.question_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(qa_frame, text="Get Answer", 
                  command=self.answer_question).grid(row=1, column=0, columnspan=2, pady=5)
        
        self.answer_text = scrolledtext.ScrolledText(qa_frame, width=80, height=10)
        self.answer_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        qa_frame.rowconfigure(2, weight=1)
        self.notebook.add(qa_frame, text="Q&A")
        
        main_frame.rowconfigure(2, weight=1)
    
    def browse_document(self):
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("Supported files", "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt")
            ]
        )
        
        if file_path:
            self.file_path_label.config(text=file_path)
            self.current_document = None
    
    def analyze_document(self):
        file_path = self.file_path_label.cget("text")
        
        if file_path == "No document selected":
            messagebox.showerror("Error", "Please select a document first")
            return
        
        if not self.document_reader.is_supported_format(file_path):
            messagebox.showerror("Error", "Unsupported file format")
            return
        
        threading.Thread(target=self._perform_analysis, args=(file_path,), daemon=True).start()
    
    def _perform_analysis(self, file_path):
        try:
            self.root.after(0, self._update_status, "Processing document...")
            
            document = self.document_reader.read_document(file_path)
            self.current_document = document
            
            analysis = self.ai_processor.analyze_document(document['text'])
            
            self.root.after(0, self._display_analysis, analysis)
            self.root.after(0, self._update_status, "Analysis complete")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, self._update_status, "Ready")
    
    def _update_status(self, message):
        self.root.title(f"AI Document Analyzer - {message}")
    
    def _display_analysis(self, analysis):
        stats_text = f"Document Analysis Results:\n\n"
        stats_text += f"Word Count: {analysis['basic_stats']['word_count']}\n"
        stats_text += f"Sentence Count: {analysis['basic_stats']['sentence_count']}\n"
        stats_text += f"Character Count: {analysis['basic_stats']['character_count']}\n"
        stats_text += f"Average Word Length: {analysis['basic_stats']['average_word_length']:.2f}\n"
        stats_text += f"Average Sentence Length: {analysis['basic_stats']['average_sentence_length']:.2f}\n"
        stats_text += f"Readability Score: {analysis['readability']}\n\n"
        stats_text += f"Sentiment: {analysis['sentiment']['sentiment']} "
        stats_text += f"(Polarity: {analysis['sentiment']['polarity']:.3f}, "
        stats_text += f"Subjectivity: {analysis['sentiment']['subjectivity']:.3f})"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        
        keywords_text = "Top Keywords:\n\n"
        for keyword, frequency in analysis['keywords']:
            keywords_text += f"{keyword}: {frequency}\n"
        
        self.keywords_text.delete(1.0, tk.END)
        self.keywords_text.insert(1.0, keywords_text)
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, analysis['summary'])
    
    def answer_question(self):
        if not self.current_document:
            messagebox.showerror("Error", "Please analyze a document first")
            return
        
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question")
            return
        
        try:
            answer = self.ai_processor.answer_question(
                self.current_document['text'], 
                question
            )
            
            result_text = f"Question: {question}\n\n"
            result_text += f"Answer: {answer['answer']}\n"
            result_text += f"Confidence: {answer['confidence']:.3f}"
            
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(1.0, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to answer question: {str(e)}")
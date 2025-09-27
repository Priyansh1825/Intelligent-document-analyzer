import tkinter as tk
from src.gui import DocumentAnalyzerGUI
import sys

def main():
    try:
        root = tk.Tk()
        app = DocumentAnalyzerGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
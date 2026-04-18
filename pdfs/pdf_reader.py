import pypdf


class PDFReader:
    def __init__(self, file_path="pdfs/experiment.pdf"):
        self.file_path = file_path
        self.reader = pypdf.PdfReader(self.file_path)
        self.pages_text=""

    def read_pdf(self):
        # Placeholder for PDF reading logic
        # In a real implementation, you would use a library like PyPDF2 or pdfplumber
        all_pages_text = []
        for page in self.reader.pages:
            all_pages_text.append(page.extract_text())
        self.pages_text = "\n".join(all_pages_text)
        return self.pages_text
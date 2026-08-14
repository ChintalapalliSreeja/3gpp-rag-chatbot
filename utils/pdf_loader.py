from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    """
    Extract text from a PDF while keeping page information.
    """

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "text": text.strip(),
                "page": page_number,
                "source": Path(pdf_path).name
            })

    return documents


if __name__ == "__main__":
    pdf_path = "data/3gpp/TS_23_501.pdf"

    documents = load_pdf(pdf_path)

    print(f"Total pages extracted: {len(documents)}")

    if documents:
        print("\nFirst page text:\n")
        print(documents[0]["text"][:3000])
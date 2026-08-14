from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_pdf


def create_chunks(pdf_path):
    # Load PDF pages
    documents = load_pdf(pdf_path)

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:
        page_chunks = text_splitter.split_text(document["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": document["page"],
                "source": document["source"]
            })

    return chunks


if __name__ == "__main__":
    pdf_path = "data/3gpp/TS_23_501.pdf"

    chunks = create_chunks(pdf_path)

    print(f"Total chunks created: {len(chunks)}")

    print("\nFirst chunk:\n")
    print(chunks[0]["text"])

    print("\nMetadata:")
    print("Source:", chunks[0]["source"])
    print("Page:", chunks[0]["page"])
import os
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed


def pdf_contains_term(file_path, search_terms_lower):
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text().lower()
            if any(term in text for term in search_terms_lower):
                return file_path
    except Exception as e:
        print(f"[Error] Could not read {file_path}: {e}")
    return None


def find_all_pdfs(base_folder):
    pdf_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def search_pdfs(base_folder, search_terms, max_workers=4):
    search_terms_lower = [term.lower() for term in search_terms]
    pdf_files = find_all_pdfs(base_folder)
    matched_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(pdf_contains_term, pdf, search_terms_lower): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                matched_files.append(result)

    return matched_files


# === Usage ===
if __name__ == "__main__":
    base_folder = "/home/yoav-levinger/Documents/private/2nd degree/Final Project/Articles"
    keywords = ["SBERT"]

    results = search_pdfs(base_folder, keywords)

    print(f"\nFound {len(results)} matching PDF(s):")
    for path in results:
        print(path)


#install first:
#pip install PyMuPDF

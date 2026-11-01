import pymupdf

pdf_path = "resume.pdf"
document = pymupdf.open(pdf_path)

text = ""

for page in document:
    text += page.get_text()

document.close()

print(text)
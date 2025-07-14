import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

# 如果你使用的是 Windows，可能需要手动设置 Tesseract 的路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_with_pymupdf(pdf_path):
    """使用 PyMuPDF 提取电子 PDF 文本"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n\n"
        if len(text.strip()) < 50:
            raise ValueError("PDF 中可能没有可提取的电子文本，尝试使用 OCR")
        return text
    except Exception as e:
        print(f"[PyMuPDF 提取失败]：{e}")
        raise

def extract_text_with_ocr(pdf_path, lang="chi_sim"):
    """使用 OCR 提取扫描版 PDF"""
    try:
        print("正在使用 OCR 模式识别中文文字（这可能需要一会儿）...")
        pages = convert_from_path(pdf_path)
        text = ""
        for i, page in enumerate(pages):
            text += pytesseract.image_to_string(page, lang=lang) + "\n\n"
        return text
    except Exception as e:
        print(f"[OCR 失败]：{e}")
        raise

def clean_text(text):
    """清洗文本：去掉页码等无用信息"""
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # 删除单独一行的页码
    return text.strip()

def pdf_to_book_txt(pdf_path, output_path="book.txt"):
    """主函数：尝试提取文本并保存为 book.txt"""
    try:
        text = extract_text_with_pymupdf(pdf_path)
        print("[✔] 使用 PyMuPDF 成功提取文本")
    except:
        text = extract_text_with_ocr(pdf_path, lang="chi_sim")
        print("[✔] 使用 OCR 成功提取文本")
    
    cleaned = clean_text(text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"[✅] 文本已保存到：{output_path}")

#path_book="data source/精神障碍诊疗规范.pdf"
#path_book="data source/DSM 5 TR.pdf"
#path_book="data source/ICD-11.pdf"
#path_book="data source/Psychiatry_7th_Edition.pdf"
#path_book="data source/Psychopathology_Second_Edition.pdf"
path_book="data source/Shorter.Oxford.Textbook.of.Psychiatry).pdf"



# 示例调用
if __name__ == "__main__":
    pdf_file =path_book  # 替换成你的 PDF 路径
    pdf_to_book_txt(pdf_file, output_path="books/book_6.txt")

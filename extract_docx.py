import zipfile
import xml.etree.ElementTree as ET

docx_path = 'c:/Projects/12_Capstone_Project/01_Problem/AI application.docx'
output_path = 'c:/Projects/12_Capstone_Project/extracted_text.txt'

with zipfile.ZipFile(docx_path, 'r') as z:
    with z.open('word/document.xml') as f:
        tree = ET.parse(f)
        root = tree.getroot()

ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

paragraphs = []
for para in root.iter('{%s}p' % ns):
    texts = []
    for t in para.iter('{%s}t' % ns):
        if t.text:
            texts.append(t.text)
    paragraphs.append(''.join(texts))

content = '\n'.join(paragraphs)

with open(output_path, 'w', encoding='utf-8') as out:
    out.write(content)

print('Done. Written to', output_path)

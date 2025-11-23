from docling.document_converter import DocumentConverter

source = "NIPS-2017-attention-is-all-you-need-Paper.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
text = result.document.export_to_markdown() # output: "## Docling Technical Report[...]"

output_path = "/Users/evier/PycharmProjects/DocumentSplit/result/docling_test_nips.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

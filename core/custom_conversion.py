import json
import logging
import time
from pathlib import Path

#from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main(input_doc_path, output_dir, doc_filename):
    logging.basicConfig(level=logging.INFO)
    

    ###########################################################################

    # The following sections contain a combination of PipelineOptions
    # and PDF Backends for various configurations.
    # Uncomment one section at the time to see the differences in the output.

    # PyPdfium without EasyOCR
    # --------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = False

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # PyPdfium with EasyOCR
    # -----------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # Docling Parse without EasyOCR
    # -------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with EasyOCR
    # ----------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["es"]
    # pipeline_options.accelerator_options = AcceleratorOptions(
    #     num_threads=4, device=AcceleratorDevice.AUTO
    # )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Docling Parse with EasyOCR (CPU only)
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.ocr_options.use_gpu = False  # <-- set this.
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract CLI
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractCliOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with ocrmac(Mac only)
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = OcrMacOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    ###########################################################################

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")


    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # # Export Text format:
    # with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
    #     fp.write(conv_result.document.export_to_text())

    # # Export Markdown format:
    # with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
    #     fp.write(conv_result.document.export_to_markdown())

    # # Export Document Tags format:
    # with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
    #     fp.write(conv_result.document.export_to_document_tokens())

def convert_one_doc(input_doc_path, output_dir, doc_filename):
    # Automatically extract output_dir and doc_filename from input_doc_path
    input_path = Path(input_doc_path)
    doc_filename = input_path.stem  # Get filename without extension
    
    # Convert data path to out path by replacing 'data' with 'out'
    input_str = str(input_path)
    output_str = input_str.replace('/data/', '/out/')
    output_path = Path(output_str)
    
    # Create output directory by removing the filename and adding the doc_filename as a folder
    output_dir = output_path.parent / doc_filename
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input path: {input_doc_path}")
    print(f"Output directory: {output_dir}")
    print(f"Document filename: {doc_filename}")
    
    main(input_doc_path, output_dir, doc_filename)

if __name__ == "__main__":
    data_folder = Path('/Users/yiminglin/Documents/Codebase/LSF/data/sample')
    
    # Scan and collect all PDFs from the data folder
    input_doc_paths = []
    for pdf_file in data_folder.rglob("*.pdf"):
        input_doc_paths.append(pdf_file)

    i = 0
    for file in input_doc_paths:
        print(file)
        print(i, len(input_doc_paths))
        i += 1
        convert_one_doc(file, data_folder, file.stem)
        break 

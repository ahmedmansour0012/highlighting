import os
import fitz
import numpy as np
from io import BytesIO
from collections import defaultdict

# Import your modules
from core.layout import layout_detect, crop_region
from core.ocr import get_ocr_object_per_page, join_ocr_texts, get_ocr_instance
from core.pdf_to_image import convert_pdf_with_pymupdf
from core.serial_extraction.extractors import find_space_separated_codes, find_classic_serials
from core.tables.detection import istable
from core.tables.crops import process_table_image
from core.tables.table_model import TableColumn, DetectedTable
from core.tables.structure import cluster_paragraphs_from_rec_columns 
from core.search.matching import exact_match_in_tables
from core.visualization.drawing import draw_mapped_boxes, map_matches_to_original_image
from core.serial_extraction.page_finder import find_pages, match_serials_to_pages
from core.serial_extraction.parsing import _split_serial


def run_pipeline_serial(pdf_path, serial_no ,output_pdf = "full_document_with_annotations.pdf"):
    print(serial_no)
    serial_parts = set()
    for serial in serial_no:
        serial_parts.update(_split_serial(serial))
        print(serial_parts)
    ocr = get_ocr_instance()
    images = convert_pdf_with_pymupdf(pdf_path = pdf_path)
    ocr_results = get_ocr_object_per_page(images,ocr)
    main_serial_parts = list(serial_parts)
    
    page_index, all_matched_components_dict = find_pages(ocr_results, main_serial_parts)
    
    selected_images = [images[i] for i in page_index if i < len(images)]
    rec_res = layout_detect(selected_images)
    tables_objs = []
    serial_to_pages = match_serials_to_pages(serial_no, all_matched_components_dict)
    print(serial_to_pages)
    # # for pag in page_index:
    # #     serial_to_pages[serial_no].add(pag)

    # Step 4: Table detection & column processing
    for page_idx, res in zip(page_index, rec_res):
        d = res.summary()
        d_sorted = sorted(d, key=lambda x: (x['box']['y1'], x['box']['x1']))
        
        # Re-classify low-confidence figures as tables if they look like tables
        for det in d_sorted:
            if det['name'] == 'figure' and det['confidence'] < 0.9:
                img_array = np.array(images[page_idx])
                cropped = crop_region(img_array, det["box"])
                if istable(cropped):
                    det['class'] = 5  # table class

        # Process actual tables
        for det in d_sorted:
            if det['class'] == 5:
                table_bbox = det["box"]
                img_array = np.array(images[page_idx])
                table_img = crop_region(img_array, table_bbox)

                col_cropsi, colum_bbox = process_table_image(table_img, ocr)
                columns = []

                for col in col_cropsi:
                    col_result = ocr.predict(col)
                    texts = col_result[0]['rec_texts'] if col_result else []
                    boxes = col_result[0]['rec_boxes'] if col_result else []
                    para_outputs, new_rec_boxes, new_rec_texts = cluster_paragraphs_from_rec_columns(boxes, texts)

                    columns.append(
                        TableColumn(bbox=colum_bbox, image=col, texts=new_rec_texts, boxes=new_rec_boxes)
                    )

                t = DetectedTable(
                    page_index=page_idx,
                    table_bbox=table_bbox,
                    table_image=table_img,
                    columns=columns
                )
                tables_objs.append(t)

    # Step 5: Match serials in tables and annotate
    for serial, page_indices in serial_to_pages.items():
        print(f"\nProcessing serial: {serial}, pages: {page_indices}")
        
        # Create a copy of images once per serial
        images_copy = images.copy()
        page_indices = [page_indices]
        # Process all pages for this serial
        for page_idx in page_indices:
            page_tables = [t for t in tables_objs if t.page_index == page_idx]
            found, all_hits, not_found = exact_match_in_tables(page_tables, serial)
            print(f"  Page {page_idx}, Found: {found}")
            
            if found:  # Only process if matches were found
                mapped_boxes = map_matches_to_original_image(all_hits)
                
                # Adjust page_index in mapped_boxes to ensure correct page
                for box in mapped_boxes:
                    box['page_index'] = page_idx
                
                processed_page_indices, annotated_images, _ = draw_mapped_boxes(
                    images_copy, mapped_boxes, out_prefix=f"serial_{serial}_page_{page_idx}"
                )
                
                # Update the images_copy with annotated images
                for idx, img in zip(processed_page_indices, annotated_images):
                    images_copy[idx] = img
            else:
                print(f"  No matches found on page {page_idx} for serial {serial}")
        
        # Step 6: Save annotated PDF (only once per serial)
        output_pdf = f"outputs/{serial}.pdf"
        doc = fitz.open()
        for img in images_copy:
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            imgdoc = fitz.open(stream=img_byte_arr.read(), filetype="png")
            pdfbytes = imgdoc.convert_to_pdf()
            doc.insert_pdf(fitz.open("pdf", pdfbytes))
        doc.save(output_pdf)
        doc.close()  # Close the document to free resources
        print(f"✅ Annotated PDF saved as: {output_pdf}")    
    return output_pdf
    
    
    
def run_pipeline_detect(pdf_path, output_pdf = "full_document_with_annotations_det.pdf"):
    """
    Runs the full document processing pipeline on a given PDF.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_pdf (str): Path to save the annotated output PDF.
    
    Returns:
        str: Path to the saved annotated PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    print(f"Processing PDF: {pdf_path}")

    ocr = get_ocr_instance()
    
    # Step 1: Convert PDF to images
    images = convert_pdf_with_pymupdf(pdf_path)
    
    # Step 2: Layout detection
    rec_res = layout_detect(images)
    
    # Step 3: OCR per page
    ocr_list = get_ocr_object_per_page(images, ocr, rec_res)
    # print(ocr_list)
    serial_to_pages = defaultdict(set)

    for page_idx, page in enumerate(ocr_list):
        for element in page:
            rec_texts = element[0]['rec_texts']
            print(rec_texts)
            rec_polys = element[0]['rec_polys']
            big_text = join_ocr_texts(rec_texts, rec_polys, y_tolerance=10)
            print(big_text)
            # Find serials
            serials = find_classic_serials(big_text)
            print(serials)
            serials_space = find_space_separated_codes(big_text)
            for s in serials + serials_space:
                serial_to_pages[s].add(page_idx)

    # Get unique pages with serials
    page_index = sorted({p for pages in serial_to_pages.values() for p in pages})
    selected_images = [images[i] for i in page_index if i < len(images)]
    selected_rec_res = [rec_res[i] for i in page_index if i < len(rec_res)]
    tables_objs = []

    # Step 4: Table detection & column processing
    for page_idx, res in zip(page_index, selected_rec_res):
        print('HERE')
        d = res.summary()
        d_sorted = sorted(d, key=lambda x: (x['box']['y1'], x['box']['x1']))
        
        # Re-classify low-confidence figures as tables if they look like tables
        for det in d_sorted:
            if det['name'] == 'figure' and det['confidence'] < 0.9:
                img_array = np.array(images[page_idx])
                cropped = crop_region(img_array, det["box"])
                if istable(cropped):
                    det['class'] = 5  # table class

        # Process actual tables
        for det in d_sorted:
            if det['class'] == 5:
                table_bbox = det["box"]
                img_array = np.array(images[page_idx])
                table_img = crop_region(img_array, table_bbox)

                col_cropsi, colum_bbox = process_table_image(table_img, ocr)
                columns = []

                for col in col_cropsi:
                    col_result = ocr.predict(col)
                    texts = col_result[0]['rec_texts'] if col_result else []
                    boxes = col_result[0]['rec_boxes'] if col_result else []
                    para_outputs, new_rec_boxes, new_rec_texts = cluster_paragraphs_from_rec_columns(boxes, texts)

                    columns.append(
                        TableColumn(bbox=colum_bbox, image=col, texts=new_rec_texts, boxes=new_rec_boxes)
                    )

                t = DetectedTable(
                    page_index=page_idx,
                    table_bbox=table_bbox,
                    table_image=table_img,
                    columns=columns
                )
                tables_objs.append(t)

    # Step 5: Match serials in tables and annotate
    for serial, page_indices in serial_to_pages.items():
        for page_idx in page_indices:
            page_tables = [t for t in tables_objs if t.page_index == page_idx]
            found, all_hits, not_found = exact_match_in_tables(page_tables, serial)
            mapped_boxes = map_matches_to_original_image(all_hits)
            
            processed_page_indices, annotated_images, _ = draw_mapped_boxes(
                images, mapped_boxes, out_prefix=f"page"
            )
            
            for idx, img in zip(processed_page_indices, annotated_images):
                images[idx] = img

    # Step 6: Save annotated PDF
    doc = fitz.open()
    for img in images:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        imgdoc = fitz.open(stream=img_byte_arr.read(), filetype="png")
        pdfbytes = imgdoc.convert_to_pdf()
        doc.insert_pdf(fitz.open("pdf", pdfbytes))
    doc.save(output_pdf)
    print(f"✅ Annotated PDF saved as: {output_pdf}")
    
    return output_pdf


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pipeline_runner import run_pipeline_serial, run_pipeline_detect
import os
import uuid
import tempfile

app = Flask(__name__)

# Output folder for final annotated PDFs
OUTPUT_FOLDER = "./outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/highlight", methods=["POST"])
def highlight():
    # ---- Check if a file was uploaded ----
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files["pdf"]
    original_filename = pdf_file.filename

    # ---- Validation ----
    if not original_filename:
        return jsonify({"error": "Empty filename"}), 400

    if not original_filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    # serial_number = request.form.get("serial_number", None)
    # detect = serial_number is None
    # Try to get serial numbers from JSON in 'data' field or form field
    serial_numbers = None
    detect = True
     # Check if there's a 'data' field with JSON content
    if 'data' in request.form:
        try:
            data = request.form['data']
            if isinstance(data, str):
                import json
                data_dict = json.loads(data)
                serial_numbers = data_dict.get('serial_numbers', [])
            else:
                serial_numbers = data.get('serial_numbers', [])
        except (json.JSONDecodeError, AttributeError):
            # If 'data' is not JSON, fall back to form field
            serial_input = request.form.get("serial_number", None)
            if serial_input:
                serial_numbers = [s.strip() for s in serial_input.split(',') if s.strip()]
    else:
        # Check regular form field
        serial_input = request.form.get("serial_number", None)
        if serial_input:
            serial_numbers = [s.strip() for s in serial_input.split(',') if s.strip()]
    
    detect = serial_numbers is None or len(serial_numbers) == 0


    # Create a secure temp file for the upload
    temp_fd = None
    temp_path = None
    try:
        # Save uploaded file to a secure temporary file
        filename = secure_filename(original_filename)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        pdf_file.save(temp_path)

        # Prepare output path
        output_filename = f"annotated_{uuid.uuid4().hex}.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # ---- Run pipeline with FILE PATH (not bytes) ----
        if detect:
            result_path ,serial_numbers  = run_pipeline_detect(temp_path, output_pdf=output_path)
        else:
            # Validate serial_number if needed (e.g., not empty)
            if not serial_numbers:
                return jsonify({"error": "serial_number is required in non-detect mode"}), 400
            result_path = run_pipeline_serial(temp_path, serial_numbers)

        # ---- Verify output file exists ----
        if not result_path or not os.path.exists(result_path):
            return jsonify({"error": "Pipeline did not produce output PDF"}), 500

        return jsonify({
            "success": True,
            "message": "PDF processed and annotated successfully.",
            "output_file": OUTPUT_FOLDER,
            "serial_number_used": serial_numbers,
            "auto_detect_mode": detect
        })

    except Exception as e:
        print(f"[ERROR] PDF processing failed: {e}")
        return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
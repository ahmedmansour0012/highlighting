# Serial Number PDF Annotator

A Flask-based web service that processes PDF files to highlight text matching specific serial numbers or automatically detects serial numbers in documents.

## Features

- **PDF Upload**: Accept PDF files via multipart form data
- **Serial Number Matching**: Highlight text matching provided serial numbers
- **Multiple Serial Numbers**: Support for processing multiple serial numbers simultaneously
- **Auto-Detection Mode**: Automatically detect serial numbers when none are provided
- **Output Generation**: Creates annotated PDFs with highlighted matches
- **RESTful API**: Simple HTTP interface for integration


## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install requirments.txt
   ```

## Usage

### API Endpoint

`POST /highlight`

### Request Format

The API accepts multipart form data with the following fields:

#### Required:
- `pdf`: PDF file to be processed

#### Optional:
- `serial_number`: Single serial number or comma-separated list of serial numbers
- `data`: JSON string containing `serial_numbers` array

### Request Examples

#### Single Serial Number (Comma-separated):
```bash
curl -X POST \
  -F "pdf=@document.pdf" \
  -F "serial_number=ABC123,XYZ789" \
  http://localhost:8000/highlight
```

#### Multiple Serial Numbers via JSON:
```bash
curl -X POST \
  -F "pdf=@document.pdf" \
  -F "data={\"serial_numbers\": [\"ABC123\", \"XYZ789\", \"DEF456\"]}" \
  http://localhost:8000/highlight
```

#### Auto-Detection Mode (No Serial Numbers):
```bash
curl -X POST \
  -F "pdf=@document.pdf" \
  http://localhost:8000/highlight
```

### Response Format

```json
{
  "success": true,
  "message": "PDF processed and annotated successfully.",
  "output_file": "/path/to/annotated_file.pdf",
  "serial_number_used": ["ABC123", "XYZ789"],
  "auto_detect_mode": false
}
```

### Error Responses

- `400`: Invalid request (missing PDF, invalid file format, etc.)
- `500`: Internal server error (pipeline failure, etc.)

## Configuration

- **Output Directory**: PDFs are saved to `./outputs` directory (created automatically)
- **Port**: Default port is 8000
- **Host**: Default host is 0.0.0.0 (accessible from network)

## Running the Service

```bash
python app.py
```

The service will start on `http://localhost:8000`

## API Modes

### 1. Manual Mode
- Provide specific serial numbers to search for
- Highlights exact matches of provided serial numbers
- Use `serial_number` field or JSON `serial_numbers` array

### 2. Auto-Detection Mode  
- No serial numbers provided
- Automatically detects serial numbers in the document
- Highlights detected serial numbers

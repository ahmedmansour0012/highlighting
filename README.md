# Table Annotation Pipeline

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install requirments.txt
   ```

2. **Place your PDF** in the working directory (e.g., `input.pdf`).

4. **Run the pipeline**:
   ```bash
   python main.py input.pdf
   ```

5. **Output**:
   - Annotated PDF: `full_document_with_annotations.pdf`
   - Debug images: `page_0.png`, `page_1.png`, etc.

---

## 📁 Project Structure

```
.
├── main.py                 # Full pipeline (run this)
├── full_document_with_annotations.pdf  # Output
└── page_*.png              # Intermediate annotated page images
```

---
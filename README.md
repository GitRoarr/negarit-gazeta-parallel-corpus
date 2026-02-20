# Negarit Gazeta Parallel Corpus Extractor

A high-accuracy tool for extracting parallel Amharic-English corpora from Ethiopian Federal Negarit Gazeta proclamations.

## Overview

This project provides a Python-based pipeline designed to handle the specific challenges of Negarit Gazeta PDFs, which often feature a Two-Column layout (Amharic on the left, English on the right) and may be scanned documents requiring robust OCR.

### Key Features

*   **Advanced Image Preprocessing**: Uses OpenCV for denoising, Otsu thresholding, and deskewing to significantly improve Tesseract OCR accuracy.
*   **Column-Aware Extraction**: Automatically splits pages into left/right columns to maintain language separation.
*   **Intelligent Segmentation**: 
    *   Amharic text is segmented based on the Ethiopic full stop (`።`).
    *   English text is segmented using NLTK's `sent_tokenize`.
*   **Sentence Alignment**: Correlates Amharic and English sentences 1:1 to create a parallel dataset.
*   **Quality Metrics**: Generates statistics on character length ratios and missing translations to help evaluate corpus quality.

## Project Structure

*   `extract_parallel_corpus_v2.py`: The main extraction script.
*   `parallel_corpus_proclamation_30_v2.csv`: The final extracted parallel dataset (Amharic ↔ English).
*   `cleaned_amharic_v2.txt` / `cleaned_english_v2.txt`: Intermediate debug files containing cleaned OCR text per page.
*   `parallel_corpus_viewer.html`: A web-based tool to visually inspect and verify the aligned pairs.

## Requirements

### System Dependencies
*   **Tesseract OCR** with Amharic language support:
    ```bash
    sudo apt install tesseract-ocr tesseract-ocr-amh
    ```
*   **Poppler** (for PDF to image conversion):
    ```bash
    sudo apt install poppler-utils
    ```

### Python Dependencies
Install required packages via pip:
```bash
pip install nltk pandas pytesseract pdf2image Pillow opencv-python-headless numpy
```

## Usage

1.  **Extract Corpus**:
    Run the script by providing the path to a Negarit Gazeta PDF:
    ```bash
    python extract_parallel_corpus_v2.py path/to/proclamation.pdf
    ```

2.  **Verify Results**:
    Open `parallel_corpus_viewer.html` in your browser to check the alignment quality of the generated CSV.

## Data Schema

The output CSV (`parallel_corpus_proclamation_30_v2.csv`) contains:
*   `id`: Unique identifier for the sentence pair.
*   `amharic`: The extracted Amharic sentence.
*   `english`: The corresponding English translation.

## Accuracy Notes

Accuracy depends heavily on the quality of the source PDF. The current version includes:
*   Regex filters to remove common OCR noise (e.g., "Federal Negarit Gazeta" headers, page numbers).
*   Unicode filtering for Amharic characters.
*   Empty/Short sentence pruning.

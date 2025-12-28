# checks-ocr

Offline check scanning utility that extracts:

- MICR line (routing/account/check numbers) using EasyOCR + OpenCV
- Printed amount using EasyOCR
- Handwritten amount using EasyOCR

Outputs a single YAML document to stdout.

## Requirements

- Python 3.11+ (tested with python.org macOS Python 3.13)

## Install

Create/activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### macOS SSL note (EasyOCR model download)

EasyOCR downloads its model weights the first time it runs. On macOS with python.org Python you may see:

`CERTIFICATE_VERIFY_FAILED`

Two common fixes:

1) Run:

- Finder -> Applications -> Python 3.13 -> Install Certificates.command

2) Or install `certifi` and run with `SSL_CERT_FILE`:

```bash
python3 -m pip install -U certifi
SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')" \
python3 scan_check.py /path/to/check.pdf
```

## Usage

The script accepts either an image (`.png`, `.jpg`) or a PDF (`.pdf`). For PDFs, the first page is rendered and processed.

```bash
python3 scan_check.py /path/to/check.pdf
```

### ROI (recommended)

To improve accuracy, specify regions-of-interest (ROI) for the printed and handwritten amount boxes.

ROI format:

`x,y,w,h` (all integers, in pixels)

Example:

```bash
python3 scan_check.py /path/to/check.pdf \
  --printed-roi 1050,350,550,140 \
  --handwritten-roi 1050,350,550,140
```

### EasyOCR model directory (optional)

To control where EasyOCR stores/loads models (useful for offline runs after the first download):

```bash
python3 scan_check.py /path/to/check.pdf \
  --easyocr-model-dir /path/to/easyocr-models
```

### GPU (optional)

If your environment supports it:

```bash
python3 scan_check.py /path/to/check.pdf --easyocr-gpu
```

## Output

The program prints a YAML document to stdout with keys:

- `input`
- `micr`
- `printed_amount`
- `handwritten_amount`

Example:

```yaml
input:
  image_path: /path/to/check.pdf
micr:
  ok: true
  raw: "..."
  digits: "..."
  routing_number: "..."
  account_number: "..."
  check_number: "..."
printed_amount:
  ok: true
  text: "..."
  amount: "..."
handwritten_amount:
  ok: true
  text: "..."
  amount: "..."
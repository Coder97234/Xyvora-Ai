import base64
import csv
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ----------------- Safe XML Helpers for DOCX / XLSX / PPTX -----------------

def _safe_parse_xml(xml_bytes: bytes) -> ET.Element:
    # Reject DTD / ENTITY declarations to prevent XXE / Billion Laughs
    head = xml_bytes[:2048].upper()
    if b"<!ENTITY" in head or b"<!DOCTYPE" in head:
        raise ValueError("XML files containing DTD/ENTITY declarations are rejected for security.")
    return ET.fromstring(xml_bytes)


def extract_pdf_sections(raw_bytes: bytes) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(raw_bytes))
    sections: List[Tuple[str, str]] = []
    for idx, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").strip()
        if txt:
            sections.append((f"Page {idx + 1}", txt))
    meta_raw = reader.metadata or {}
    meta = {
        "format": "pdf",
        "page_count": len(reader.pages),
        "author": str(getattr(meta_raw, "author", "") or ""),
        "pdf_title": str(getattr(meta_raw, "title", "") or ""),
    }
    return sections, meta


def extract_docx_sections(raw_bytes: bytes) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    sections: List[Tuple[str, str]] = []
    para_count = 0
    table_count = 0

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Invalid DOCX archive: missing word/document.xml")
        root = _safe_parse_xml(zf.read("word/document.xml"))

    ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(f"{ns_w}body")
    if body is None:
        return [], {"format": "docx", "paragraphs": 0, "tables": 0}

    current_heading = "Document Body"
    current_lines: List[str] = []

    def flush() -> None:
        if current_lines:
            sections.append((current_heading, "\n\n".join(current_lines)))
            current_lines.clear()

    for elem in body:
        if elem.tag == f"{ns_w}p":
            texts = [t.text for t in elem.iter(f"{ns_w}t") if t.text]
            line = "".join(texts).strip()
            if not line:
                continue
            para_count += 1
            # Detect Heading style
            pPr = elem.find(f"{ns_w}pPr")
            style_val = ""
            if pPr is not None:
                pStyle = pPr.find(f"{ns_w}pStyle")
                if pStyle is not None:
                    style_val = pStyle.attrib.get(f"{ns_w}val", "")
            if "heading" in style_val.lower() or (len(line) < 80 and style_val.lower() == "title"):
                flush()
                current_heading = line[:80]
            else:
                current_lines.append(line)
        elif elem.tag == f"{ns_w}tbl":
            table_count += 1
            rows_md: List[str] = []
            for r_idx, tr in enumerate(elem.findall(f"{ns_w}tr")):
                cells = []
                for tc in tr.findall(f"{ns_w}tc"):
                    c_txt = "".join(t.text for t in tc.iter(f"{ns_w}t") if t.text).strip()
                    cells.append(c_txt.replace("|", "/"))
                if cells:
                    rows_md.append("| " + " | ".join(cells) + " |")
                    if r_idx == 0:
                        rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if rows_md:
                current_lines.append("\n".join(rows_md))

    flush()
    return sections, {"format": "docx", "paragraphs": para_count, "tables": table_count}


def extract_xlsx_sections(raw_bytes: bytes) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    sections: List[Tuple[str, str]] = []
    total_rows = 0

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = set(zf.namelist())
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            ss_root = _safe_parse_xml(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.iter():
                if si.tag.endswith("}si") or si.tag == "si":
                    txt = "".join(t.text for t in si.iter() if (t.tag.endswith("}t") or t.tag == "t") and t.text)
                    shared_strings.append(txt)

        sheet_names: List[str] = []
        if "xl/workbook.xml" in names:
            wb_root = _safe_parse_xml(zf.read("xl/workbook.xml"))
            for sh in wb_root.iter():
                if sh.tag.endswith("}sheet") or sh.tag == "sheet":
                    sheet_names.append(sh.attrib.get("name", f"Sheet{len(sheet_names)+1}"))

        sheet_files = sorted(n for n in names if re.match(r"^xl/worksheets/sheet\d+\.xml$", n))
        for idx, sf in enumerate(sheet_files):
            s_label = sheet_names[idx] if idx < len(sheet_names) else f"Sheet {idx + 1}"
            s_root = _safe_parse_xml(zf.read(sf))
            rows_out: List[List[str]] = []
            for row_el in s_root.iter():
                if row_el.tag.endswith("}row") or row_el.tag == "row":
                    vals: List[str] = []
                    for c_el in row_el:
                        if not (c_el.tag.endswith("}c") or c_el.tag == "c"):
                            continue
                        c_type = c_el.attrib.get("t", "")
                        v_text = ""
                        for child in c_el:
                            if child.tag.endswith("}v") or child.tag == "v":
                                v_text = child.text or ""
                            elif child.tag.endswith("}is") or child.tag == "is":
                                v_text = "".join(t.text for t in child.iter() if t.text)
                        if c_type == "s" and v_text.isdigit():
                            s_idx = int(v_text)
                            if 0 <= s_idx < len(shared_strings):
                                v_text = shared_strings[s_idx]
                        vals.append(v_text.strip())
                    if any(vals):
                        rows_out.append(vals)
                        total_rows += 1

            if rows_out:
                lines = []
                header = rows_out[0]
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for r in rows_out[1:250]:
                    padded = r + [""] * max(0, len(header) - len(r))
                    lines.append("| " + " | ".join(padded[: len(header)]) + " |")
                sections.append((f"Sheet: {s_label}", "\n".join(lines)))

    return sections, {"format": "xlsx", "sheets": len(sections), "rows": total_rows}


def extract_pptx_sections(raw_bytes: bytes) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    sections: List[Tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = zf.namelist()
        slide_files = sorted(
            (n for n in names if re.match(r"^ppt/slides/slide\d+\.xml$", n)),
            key=lambda s: int(re.findall(r"\d+", s)[0]),
        )
        for idx, sf in enumerate(slide_files):
            s_root = _safe_parse_xml(zf.read(sf))
            runs = [
                el.text.strip()
                for el in s_root.iter()
                if (el.tag.endswith("}t") or el.tag == "t") and el.text and el.text.strip()
            ]
            notes_path = f"ppt/notesSlides/notesSlide{idx + 1}.xml"
            if notes_path in names:
                n_root = _safe_parse_xml(zf.read(notes_path))
                note_runs = [
                    el.text.strip()
                    for el in n_root.iter()
                    if (el.tag.endswith("}t") or el.tag == "t") and el.text and el.text.strip()
                ]
                if note_runs:
                    runs.append("Speaker Notes: " + " ".join(note_runs))
            if runs:
                sections.append((f"Slide {idx + 1}", "\n".join(runs)))

    return sections, {"format": "pptx", "slide_count": len(slide_files)}


def extract_csv_sections(text: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    reader = list(csv.reader(io.StringIO(text)))
    if not reader:
        return [], {"format": "csv", "rows": 0, "columns": []}
    headers = [h.strip() or f"col_{i+1}" for i, h in enumerate(reader[0])]
    data_rows = [r for r in reader[1:] if any(c.strip() for c in r)]

    # Compute numeric column summaries automatically
    col_stats: List[str] = []
    for c_idx, h in enumerate(headers):
        nums: List[float] = []
        for r in data_rows:
            if c_idx < len(r):
                val_s = r[c_idx].strip().replace(",", "")
                try:
                    nums.append(float(val_s))
                except ValueError:
                    pass
        if len(nums) >= 2:
            mean_v = sum(nums) / len(nums)
            col_stats.append(f"- Column `{h}` (n={len(nums)}): min={min(nums):g}, max={max(nums):g}, mean={mean_v:.4g}")

    sections: List[Tuple[str, str]] = []
    summary_txt = f"Columns ({len(headers)}): {', '.join(headers)}\nTotal Data Rows: {len(data_rows)}"
    if col_stats:
        summary_txt += "\nNumeric Column Profiles:\n" + "\n".join(col_stats)
    sections.append(("CSV Schema & Summary", summary_txt))

    batch_size = 30
    for start in range(0, min(len(data_rows), 300), batch_size):
        batch = data_rows[start : start + batch_size]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in batch:
            padded = r + [""] * max(0, len(headers) - len(r))
            lines.append("| " + " | ".join(c.replace("|", "/") for c in padded[: len(headers)]) + " |")
        sections.append((f"Rows {start + 1}–{start + len(batch)}", "\n".join(lines)))

    return sections, {"format": "csv", "rows": len(data_rows), "columns": headers}


def extract_code_sections(text: str, filename: str, ext: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    chunk_lines = 55
    for start in range(0, len(lines), chunk_lines):
        end = min(len(lines), start + chunk_lines)
        snippet = "\n".join(lines[start:end]).strip()
        if snippet:
            sections.append((f"{filename} (Lines {start + 1}–{end})", snippet))
    return sections, {"format": "source_code", "extension": ext, "line_count": len(lines)}


def extract_markdown_or_text_sections(text: str, is_markdown: bool) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    if not is_markdown:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [(f"Paragraph {i+1}", p) for i, p in enumerate(paras)], {"format": "text", "paragraphs": len(paras)}

    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"
    buf: List[str] = []
    for line in text.splitlines():
        hm = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if hm:
            if buf and "\n".join(buf).strip():
                sections.append((current_heading, "\n".join(buf).strip()))
            current_heading = hm.group(2).strip()[:80]
            buf = []
        else:
            buf.append(line)
    if buf and "\n".join(buf).strip():
        sections.append((current_heading, "\n".join(buf).strip()))
    return sections, {"format": "markdown", "sections": len(sections)}


def parse_document_bytes(
    filename: str, raw_bytes: bytes, mime_type: str = ""
) -> Tuple[str, List[Tuple[str, str]], Dict[str, Any]]:
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File '{filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf" or mime_type == "application/pdf":
        sections, meta = extract_pdf_sections(raw_bytes)
    elif ext == ".docx" or "wordprocessingml" in mime_type:
        sections, meta = extract_docx_sections(raw_bytes)
    elif ext == ".xlsx" or "spreadsheetml" in mime_type:
        sections, meta = extract_xlsx_sections(raw_bytes)
    elif ext == ".pptx" or "presentationml" in mime_type:
        sections, meta = extract_pptx_sections(raw_bytes)
    elif ext in (".csv", ".tsv") or mime_type in ("text/csv", "text/tab-separated-values"):
        text = raw_bytes.decode("utf-8", errors="replace")
        sections, meta = extract_csv_sections(text)
    elif ext in (".json", ".jsonl") or mime_type == "application/json":
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text)
            pretty = json.dumps(obj, indent=2)
            top_keys = list(obj.keys())[:25] if isinstance(obj, dict) else f"Array[{len(obj)}]"
            sections = [("JSON Structure", f"Top-level keys/shape: {top_keys}\n\n{pretty[:12000]}")]
            meta = {"format": "json", "shape": str(top_keys)}
        except Exception:
            sections, meta = extract_markdown_or_text_sections(text, False)
    elif ext in {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".java",
        ".cpp", ".cc", ".c", ".h", ".hpp", ".cs", ".luau", ".lua", ".go",
        ".rs", ".sql", ".sh", ".yaml", ".yml", ".toml",
    }:
        text = raw_bytes.decode("utf-8", errors="replace")
        sections, meta = extract_code_sections(text, filename, ext)
    elif ext in (".md", ".mdx", ".markdown"):
        text = raw_bytes.decode("utf-8", errors="replace")
        sections, meta = extract_markdown_or_text_sections(text, True)
    else:
        # Reject binary executables/archives disguised as text
        if b"\x00" in raw_bytes[:1024]:
            raise ValueError(
                f"Unsupported binary file format '{ext or mime_type}'. Supported formats: PDF, DOCX, XLSX, PPTX, CSV, JSON, Markdown, TXT, Source Code, and Images."
            )
        text = raw_bytes.decode("utf-8", errors="replace")
        sections, meta = extract_markdown_or_text_sections(text, False)

    full_text = "\n\n".join(f"[{lbl}]\n{body}" for lbl, body in sections if body.strip())
    if not full_text.strip():
        raise ValueError(f"No readable text content could be extracted from '{filename}'.")
    return full_text, sections, meta


# ----------------- Image Inspection & OCR / Vision Preparation -----------------

def inspect_image_bytes(
    filename: str, raw_bytes: bytes, mime_type: str = ""
) -> Dict[str, Any]:
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds 10 MB upload limit.")
    if len(raw_bytes) < 16:
        raise ValueError("Invalid or truncated image file.")

    width: Optional[int] = None
    height: Optional[int] = None
    detected_mime = mime_type or "image/png"
    embedded_texts: List[str] = []

    # 1. PNG header + tEXt / iTXt chunk extraction
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_mime = "image/png"
        if len(raw_bytes) >= 24:
            width, height = struct.unpack(">II", raw_bytes[16:24])
        pos = 8
        while pos + 8 <= len(raw_bytes):
            try:
                length, chunk_type = struct.unpack(">I4s", raw_bytes[pos : pos + 8])
                chunk_data = raw_bytes[pos + 8 : pos + 8 + length]
                if chunk_type == b"tEXt":
                    decoded = chunk_data.decode("latin-1", errors="ignore").replace("\x00", ": ")
                    if decoded.strip():
                        embedded_texts.append(decoded.strip()[:300])
                elif chunk_type == b"IEND":
                    break
                pos += 12 + length
            except Exception:
                break

    # 2. GIF header
    elif raw_bytes[:6] in (b"GIF87a", b"GIF89a"):
        detected_mime = "image/gif"
        width, height = struct.unpack("<HH", raw_bytes[6:10])

    # 3. JPEG SOF0/SOF2 marker scan
    elif raw_bytes.startswith(b"\xff\xd8"):
        detected_mime = "image/jpeg"
        idx = 2
        while idx + 9 < len(raw_bytes):
            if raw_bytes[idx] != 0xFF:
                idx += 1
                continue
            marker = raw_bytes[idx + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                height, width = struct.unpack(">HH", raw_bytes[idx + 5 : idx + 9])
                break
            if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                idx += 2
                continue
            seg_len = struct.unpack(">H", raw_bytes[idx + 2 : idx + 4])[0]
            if seg_len < 2:
                break
            idx += 2 + seg_len

    # 4. WEBP RIFF header
    elif raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        detected_mime = "image/webp"
        if raw_bytes[12:16] == b"VP8X" and len(raw_bytes) >= 30:
            w_minus_1 = int.from_bytes(raw_bytes[24:27], "little")
            h_minus_1 = int.from_bytes(raw_bytes[27:30], "little")
            width, height = w_minus_1 + 1, h_minus_1 + 1

    # 5. SVG XML diagram text extraction!
    elif b"<svg" in raw_bytes[:1024].lower():
        detected_mime = "image/svg+xml"
        try:
            svg_root = _safe_parse_xml(raw_bytes)
            for el in svg_root.iter():
                if (el.tag.endswith("}text") or el.tag == "text" or el.tag.endswith("}title") or el.tag == "title") and el.text:
                    if el.text.strip():
                        embedded_texts.append(el.text.strip())
        except Exception:
            pass

    # Optional system OCR via tesseract if installed on host
    ocr_text = ""
    ocr_engine = "none"
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin and detected_mime in ("image/png", "image/jpeg", "image/webp"):
        ext = ".png" if "png" in detected_mime else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                [tesseract_bin, tmp_path, "stdout"],
                capture_output=True,
                text=True,
                timeout=6.0,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                ocr_text = proc.stdout.strip()[:4000]
                ocr_engine = "tesseract"
        except Exception:
            pass
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    combined_extracted = "\n".join(filter(None, [ocr_text] + embedded_texts)).strip()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    # For SVG, browser + vision models vary; keep data URL accurate
    data_url = f"data:{detected_mime};base64,{b64}"

    return {
        "filename": filename,
        "mime_type": detected_mime,
        "size_bytes": len(raw_bytes),
        "width": width,
        "height": height,
        "ocr_engine": ocr_engine,
        "extracted_text": combined_extracted,
        "data_url": data_url,
    }

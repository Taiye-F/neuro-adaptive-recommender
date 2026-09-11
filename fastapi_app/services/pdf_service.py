# pdf_service.py
import io
import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


class PDFService:
    @staticmethod
    def generate_clinical_report_html(context: dict) -> str:
        template = jinja_env.get_template("clinical_report.html")
        return template.render(context)

    @staticmethod
    def generate_clinical_report_pdf(context: dict) -> bytes:
        html_content = PDFService.generate_clinical_report_html(context)
        log.info("Compiling clinical report into PDF via WeasyPrint...")
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class TDRPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(
            0,
            10,
            "Propuesta Metodologica y Tecnica - Treball de Recerca (TDR)",
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C",
        )
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0,
            10,
            f"Pagina {self.page_no()}",
            border=0,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(230, 230, 230)
        self.cell(
            0,
            10,
            title,
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="L",
            fill=True,
        )
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 7, body)
        self.ln()


def create_tdr_document():
    pdf = TDRPDF()
    pdf.add_page()

    pdf.chapter_title("1. Introduccion al Proyecto")
    pdf.chapter_body(
        "Este proyecto consiste en el desarrollo de un chatbot de Inteligencia "
        "Artificial de ejecucion local basado en modelos LLM mediante Ollama. "
        "La innovacion reside en reducir dependencias externas, mantener privacidad "
        "y combinar memoria SQLite con busqueda web en tiempo real."
    )

    pdf.chapter_title("2. Metodologia de Recogida de Datos")
    pdf.chapter_body(
        "Se propone una metodologia mixta: logs automaticos en chatbot_memory.db, "
        "pruebas de interaccion con usuarios y evaluacion cualitativa de utilidad. "
        "Los datos personales se procesan localmente en el equipo del investigador."
    )

    pdf.chapter_title("3. Variables de Medicion")
    pdf.chapter_body(
        "Variables independientes: uso de busqueda web, modelo local seleccionado "
        "y cantidad de recuerdos en memoria.\n"
        "Variables dependientes: latencia, precision factual, calidad de respuesta "
        "y capacidad de recuperar preferencias o datos guardados."
    )

    pdf.chapter_title("4. Analisis de la Arquitectura")
    pdf.chapter_body(
        "La arquitectura combina memoria persistente, recuperacion de contexto, "
        "busqueda web, extraccion HTML e ingenieria de prompts para reducir el "
        "impacto del conocimiento estatico del modelo local."
    )

    pdf.chapter_title("5. Organizacion de la Muestra")
    pdf.chapter_body(
        "La muestra puede dividirse en pruebas internas, beta-testers tecnicos y "
        "usuarios no tecnicos. Cada grupo evalua precision, rapidez y naturalidad."
    )

    pdf.chapter_title("6. Limitaciones del Proyecto")
    pdf.chapter_body(
        "El rendimiento depende del hardware local, la ventana de contexto del "
        "modelo y la disponibilidad de las paginas web consultadas."
    )

    pdf.chapter_title("7. Ejemplo de Tabla de Recogida")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(15, 10, "ID", 1)
    pdf.cell(60, 10, "Pregunta", 1)
    pdf.cell(30, 10, "Uso Web", 1)
    pdf.cell(30, 10, "Latencia", 1)
    pdf.cell(30, 10, "Calidad", 1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    rows = [
        ("01", "Noticias IA hoy", "SI", "14.2s", "4/5"),
        ("02", "Cual es mi nombre", "NO", "1.5s", "5/5"),
    ]
    for row in rows:
        pdf.cell(15, 8, row[0], 1)
        pdf.cell(60, 8, row[1], 1)
        pdf.cell(30, 8, row[2], 1)
        pdf.cell(30, 8, row[3], 1)
        pdf.cell(30, 8, row[4], 1)
        pdf.ln()
    pdf.ln(10)

    pdf.chapter_title("8. Pregunta de Investigacion e Hipotesis")
    pdf.chapter_body(
        "Pregunta: Puede un modelo local ofrecer respuestas actuales mediante "
        "tecnicas de memoria y busqueda web sin depender de APIs de pago?\n\n"
        "Hipotesis: La integracion de busqueda web y memoria local permite mejorar "
        "la precision en datos de actualidad manteniendo privacidad del usuario."
    )

    output_path = "Guia_TDR_IA_Local.pdf"
    pdf.output(output_path)
    print(f"PDF generado con exito: {output_path}")


if __name__ == "__main__":
    create_tdr_document()

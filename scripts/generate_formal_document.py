from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "TulongText_PH_Formal_Project_Document.pdf"

NAVY = "#062A5E"
BLUE = "#0F4CC9"
RED = "#D91E36"
MUTED = "#4B5563"
LIGHT = "#EEF5FF"
LINE = "#D8E3F3"


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def wrap(text: str, width: int = 96) -> str:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


class Doc:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.fig = None
        self.ax = None
        self.y = 0.0
        self.page = 0

    def new_page(self, title: str | None = None):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.page += 1
        self.fig, self.ax = plt.subplots(figsize=(8.27, 11.69))
        self.ax.set_axis_off()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none"))
        self.ax.add_patch(Rectangle((0.055, 0.925), 0.89, 0.003, facecolor=RED, edgecolor="none"))
        self.y = 0.89
        if title:
            self.ax.text(0.075, self.y, title, fontsize=17, fontweight="bold", color=NAVY, va="top")
            self.y -= 0.047

    def finish(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)

    def ensure(self, need: float, title: str = "TulongText PH Project Document"):
        if self.y - need < 0.08:
            self.new_page(title)

    def heading(self, text: str):
        self.ensure(0.065)
        self.ax.text(0.075, self.y, text, fontsize=12.5, fontweight="bold", color=NAVY, va="top")
        self.y -= 0.028

    def para(self, text: str, width: int = 94, size: float = 9.7, color: str = "#111827"):
        body = wrap(text, width)
        lines = body.count("\n") + 1
        height = lines * 0.017 + 0.018
        self.ensure(height)
        self.ax.text(0.075, self.y, body, fontsize=size, color=color, va="top", linespacing=1.35)
        self.y -= height

    def bullets(self, items: list[str], width: int = 90):
        for item in items:
            body = wrap(item, width)
            lines = body.count("\n") + 1
            self.ensure(lines * 0.017 + 0.01)
            self.ax.text(0.092, self.y, u"\u2022 " + body.replace("\n", "\n  "), fontsize=9.5, color="#111827", va="top", linespacing=1.3)
            self.y -= lines * 0.017 + 0.006
        self.y -= 0.006

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float], row_h: float = 0.048):
        x0 = 0.075
        table_w = 0.85
        nrows = len(rows) + 1
        self.ensure(nrows * row_h + 0.02)
        x = x0
        self.ax.add_patch(Rectangle((x0, self.y - row_h), table_w, row_h, facecolor=LIGHT, edgecolor=LINE, linewidth=0.8))
        for i, h in enumerate(headers):
            w = widths[i] * table_w
            self.ax.text(x + 0.008, self.y - 0.014, h, fontsize=8.7, fontweight="bold", color=NAVY, va="top")
            x += w
        self.y -= row_h
        for r, row in enumerate(rows):
            x = x0
            fill = "#FFFFFF" if r % 2 == 0 else "#F8FBFF"
            self.ax.add_patch(Rectangle((x0, self.y - row_h), table_w, row_h, facecolor=fill, edgecolor=LINE, linewidth=0.6))
            for i, cell in enumerate(row):
                w = widths[i] * table_w
                txt = wrap(cell, max(18, int(w * 115)))
                self.ax.text(x + 0.008, self.y - 0.012, txt, fontsize=7.7, color="#111827", va="top", linespacing=1.12)
                x += w
            self.y -= row_h
        self.y -= 0.018


def add_cover(doc: Doc):
    doc.new_page(None)
    ax = doc.ax
    ax.add_patch(Rectangle((0.055, 0.89), 0.89, 0.008, facecolor=NAVY, edgecolor="none"))
    ax.text(0.5, 0.805, "TulongText PH: Filipino-English Disaster Post", fontsize=18, fontweight="bold", color=NAVY, ha="center")
    ax.text(0.5, 0.765, "Classification and Urgency Triage System", fontsize=18, fontweight="bold", color=NAVY, ha="center")
    ax.text(0.5, 0.70, "CCS 229 - Natural Language Processing", fontsize=12, color="#111827", ha="center")
    ax.text(0.5, 0.665, "Final Project", fontsize=12, color="#111827", ha="center")
    ax.text(0.5, 0.62, "Submitted by:", fontsize=10, fontweight="bold", color=NAVY, ha="center", va="top")
    students = "Baladjay, Aser Jr.\nDorado, Louise Marielle\nMontenegro, Karlo Roel\nPontillas, Steven Ken"
    ax.text(0.5, 0.575, students, fontsize=10, color="#111827", ha="center", va="top", linespacing=1.35)
    ax.text(0.5, 0.43, "BSCS 3A AI", fontsize=10, color="#111827", ha="center")
    ax.text(0.5, 0.36, "Submitted to:", fontsize=10, fontweight="bold", color=NAVY, ha="center")
    ax.text(0.5, 0.325, "John Cristopher Mateo", fontsize=10.5, color="#111827", ha="center")
    ax.text(0.5, 0.295, "CCS 229 - Instructor", fontsize=10, color="#111827", ha="center")
    ax.text(0.5, 0.23, "May 22, 2026", fontsize=10, color="#111827", ha="center")


def add_references(doc: Doc):
    doc.new_page("References")
    refs = [
        "HFAbrar. (n.d.). Disaster response messages [Data set]. Hugging Face. https://huggingface.co/datasets/HFAbrar/disaster_response_messages",
        "Hugging Face. (n.d.). Transformers documentation. https://huggingface.co/docs/transformers/",
        "Olteanu, A., Vieweg, S., & Castillo, C. (2015). What to expect when the unexpected happens: Social media communications across crises. Proceedings of the ACM Conference on Computer Supported Cooperative Work & Social Computing.",
        "Pallets. (n.d.). Flask documentation. https://flask.palletsprojects.com/",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "QCRI. (n.d.). CrisisBench-all-lang [Data set]. Hugging Face. https://huggingface.co/datasets/QCRI/CrisisBench-all-lang",
        "QCRI. (n.d.). CrisisMMD [Data set]. Hugging Face. https://huggingface.co/datasets/QCRI/CrisisMMD",
        "QCRI. (n.d.). HumAID-events [Data set]. Hugging Face. https://huggingface.co/datasets/QCRI/HumAID-events",
        "SEACrowd. (n.d.). Typhoon Yolanda tweets [Data set]. Hugging Face. https://huggingface.co/datasets/SEACrowd/typhoon_yolanda_tweets",
        "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., ... Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.",
    ]
    doc.para("The following sources are formatted in APA style. Each entry is placed on a separate line for readability.", width=92, color=MUTED)
    for ref in refs:
        text = wrap(ref, 86)
        lines = text.count("\n") + 1
        doc.ensure(lines * 0.019 + 0.025, "References")
        doc.ax.text(0.095, doc.y, text, fontsize=8.8, color="#111827", va="top", linespacing=1.28, fontstyle="italic")
        doc.y -= lines * 0.0185 + 0.022


def build():
    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT) as pdf:
        doc = Doc(pdf)
        add_cover(doc)

        doc.new_page("Project Overview")
        doc.para("TulongText PH is a Filipino-English disaster relief post classification and urgency triage system. It uses Natural Language Processing to classify noisy social media-style disaster posts into humanitarian response categories, estimate urgency through rule-based triage, and identify whether a post is actionable or non-actionable.")
        doc.heading("Project Objectives")
        doc.bullets([
            "Classify disaster-related posts into humanitarian categories such as rescue needs, medical concerns, evacuation, infrastructure damage, warnings, donations, general updates, and non-humanitarian content.",
            "Compare a traditional baseline model against a locally fine-tuned multilingual transformer model.",
            "Provide a browser-based demo that allows users to enter a post and view the predicted category, urgency level, confidence score, actionability label, cleaned text, and top predictions.",
            "Document the dataset sources, preprocessing steps, model evaluation, limitations, and future improvements in a reproducible final project format.",
        ])
        doc.heading("Target Users")
        doc.para("The system is intended for LGUs, disaster-response teams, volunteers, and social media monitoring groups that need faster triage support during emergencies.")

        doc.new_page("Introduction and Problem Statement")
        doc.heading("Introduction")
        doc.para("During disasters, people often share urgent information online before it reaches formal reporting channels. These posts may mention stranded families, injuries, damaged roads, evacuation centers, missing people, donation needs, or safety warnings. However, disaster posts are usually informal, multilingual, emotionally written, and mixed with irrelevant content.")
        doc.heading("Problem Statement")
        doc.para("Manual sorting of disaster-related social media posts is slow and difficult at scale. In Philippine disaster contexts, messages may be written in English, Filipino, or Taglish, and many posts contain abbreviations, incomplete sentences, hashtags, mentions, links, and noisy phrasing. TulongText PH addresses this by automatically organizing posts into response-oriented categories and urgency levels.")
        doc.heading("Significance in NLP")
        doc.para("This project is appropriate for NLP because it requires text preprocessing, tokenization, feature extraction, supervised text classification, model comparison, confidence scoring, and interpretation of noisy multilingual text.")

        doc.new_page("Datasets and Label Mapping")
        doc.heading("Dataset Sources")
        doc.table(
            ["Dataset", "Purpose in Project", "Notes"],
            [
                ["CrisisLexT26", "Disaster tweet source and crisis context support", "Contains crisis-event tweet labels used for humanitarian classification work."],
                ["QCRI HumAID-events", "Humanitarian category support", "Used for category-aligned disaster information types."],
                ["QCRI CrisisBench-all-lang", "Multilingual crisis classification source", "Used to broaden disaster and language coverage."],
                ["SEACrowd Typhoon Yolanda Tweets", "Philippine relevance support", "Sentiment-oriented data used as local context support rather than primary category training."],
                ["Disaster Response Messages", "Dataset expansion", "Adds request, aid, medical, rescue, shelter, infrastructure, and weather-related labels."],
                ["QCRI CrisisMMD", "Optional expansion source", "Adds disaster tweet text and humanitarian/damage-related examples with duplicate checks."],
            ],
            [0.27, 0.34, 0.39],
            row_h=0.066,
        )
        doc.heading("Final Category and Urgency Mapping")
        doc.table(
            ["Final Label", "Urgency", "Actionability", "Description"],
            [
                ["rescue_or_urgent_needs", "Critical", "Actionable", "Requests for rescue, urgent help, stranded-person response, or immediate needs."],
                ["medical_or_casualties", "Critical", "Actionable", "Injuries, deaths, medical support, hospitals, medicines, or health concerns."],
                ["evacuation_or_displacement", "High", "Actionable", "Evacuation centers, displaced residents, shelter, relocation, and refugee needs."],
                ["infrastructure_damage", "High", "Actionable", "Roads, bridges, power lines, buildings, communications, and utilities damage."],
                ["warnings_or_advice", "Moderate", "Actionable", "Alerts, advisories, safety instructions, weather hazards, and caution messages."],
                ["donation_or_volunteering", "Moderate", "Actionable", "Relief goods, donations, volunteer calls, food, water, clothing, and supplies."],
                ["general_update", "Low", "Non-actionable", "Relevant disaster updates, reactions, sympathy, or non-urgent information."],
                ["not_humanitarian", "Low", "Non-actionable", "Unrelated, promotional, or non-disaster content."],
            ],
            [0.29, 0.13, 0.17, 0.41],
            row_h=0.061,
        )

        doc.new_page("Preprocessing and Model Development")
        doc.heading("Preprocessing Pipeline")
        doc.bullets([
            "Normalize text by lowercasing and cleaning unnecessary whitespace.",
            "Remove or simplify URLs, mentions, and noisy social media artifacts.",
            "Preserve meaningful disaster terms, hashtags, barangay names, and local expressions when possible.",
            "Tokenize text for display and for model input.",
            "Deduplicate normalized text to reduce train-test leakage and repeated social media content.",
            "Map original dataset labels into the final eight humanitarian categories.",
        ])
        doc.heading("Models Implemented")
        doc.table(
            ["Model", "Approach", "Role"],
            [
                ["Baseline", "TF-IDF word bigram features with calibrated LinearSVC", "Stable, fast, interpretable comparison model."],
                ["Transformer", "Locally fine-tuned multilingual DistilBERT classifier", "Context-aware model using pretrained multilingual representations."],
                ["Actionability Classifier", "Binary TF-IDF classifier", "Secondary triage label identifying actionable vs non-actionable posts."],
            ],
            [0.24, 0.43, 0.33],
            row_h=0.063,
        )
        doc.para("The project does not use GPT models, third-party prediction APIs, or external classifier services. The transformer uses a locally fine-tuned pretrained DistilBERT model, which is acceptable because predictions run locally through the project backend.")

        doc.new_page("Evaluation Results")
        doc.heading("Model Evaluation Summary")
        doc.table(
            ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],
            [
                ["Baseline TF-IDF + LinearSVC", "0.7600", "0.7654", "0.7733", "0.7689", "0.7576"],
                ["Multilingual DistilBERT Transformer", "0.6661", "0.6600", "0.6661", "0.6608", "0.6636"],
                ["Actionability Binary Classifier", "0.8683", "0.8535", "0.8690", "0.8596", "0.8697"],
            ],
            [0.32, 0.13, 0.15, 0.14, 0.13, 0.13],
            row_h=0.055,
        )
        doc.heading("Feature Comparison")
        doc.table(
            ["Feature", "Baseline", "Transformer"],
            [
                ["Training speed", "Fast and practical for local machines", "Slower and more compute-heavy"],
                ["Interpretability", "Clearer through TF-IDF feature behavior", "Harder to explain directly"],
                ["Context handling", "Limited to word/phrase patterns", "Better contextual representation"],
                ["Final project reliability", "Strongest saved fine-grained result", "Useful comparison and future improvement path"],
            ],
            [0.28, 0.36, 0.36],
            row_h=0.06,
        )
        doc.heading("Interpretation")
        doc.para("The baseline outperformed the saved transformer because the dataset is relatively noisy, class labels come from multiple corpora, and the transformer was limited by local training time. The binary actionability classifier achieved higher scores because it solves an easier triage task with fewer labels.")

        doc.new_page("System Architecture and Web App")
        doc.heading("Architecture")
        doc.bullets([
            "Frontend: HTML, CSS, and JavaScript dashboard for entering posts and viewing predictions.",
            "Backend: Python Flask application with Flask-CORS support.",
            "Model layer: saved baseline, transformer, and actionability model artifacts loaded locally.",
            "API layer: /api/health, /api/models, /api/predict, and /api/batch_predict.",
        ])
        doc.heading("Web App Features")
        doc.bullets([
            "Textarea input for disaster or social media posts.",
            "Model selector for choosing the baseline or transformer model.",
            "Prediction results showing category, urgency level, actionability, confidence, cleaned text, token count, and top predictions.",
            "Demo example cards for quick testing during presentation.",
            "Prediction category guide showing all supported labels and urgency levels.",
        ])
        doc.heading("Functional Requirements")
        doc.bullets([
            "The app must accept English, Filipino, and Taglish-style disaster posts.",
            "The app must return a predicted humanitarian category, urgency level, confidence score, and actionability label.",
            "The app must show preprocessing information for transparency.",
            "The API must validate empty input and return a friendly error.",
        ])

        doc.new_page("Testing, Limitations, and Conclusion")
        doc.heading("Testing and Validation")
        doc.bullets([
            "Verified that processed CSV files load successfully and contain required text and label columns.",
            "Verified that baseline, transformer, and actionability model artifacts can be loaded by the Flask backend.",
            "Tested /api/health, /api/models, /api/predict, and /api/batch_predict.",
            "Tested demo examples for rescue, medical, evacuation, infrastructure damage, and not-humanitarian cases.",
        ])
        doc.heading("Current Limitations")
        doc.bullets([
            "Urgency is rule-based and derived from the predicted category rather than learned as a separate multiclass model.",
            "Location extraction is not yet implemented.",
            "Taglish and Filipino-labeled data are still limited compared with English data.",
            "Transformer performance is constrained by local compute and interrupted long training runs.",
        ])
        doc.heading("Future Improvements")
        doc.bullets([
            "Add location extraction for barangays, cities, and landmarks.",
            "Create a batch triage dashboard with CSV upload and export.",
            "Improve explainability with highlighted keywords and confidence explanations.",
            "Train a stronger transformer using better compute, checkpointing, and cleaner Filipino/Taglish data.",
        ])
        doc.heading("Conclusion")
        doc.para("TulongText PH demonstrates how NLP can support humanitarian triage by organizing noisy disaster posts into useful categories, urgency levels, and actionability labels. The system is not intended to replace human responders, but it can help reduce manual sorting time and make crisis information easier to prioritize.")

        add_references(doc)
        doc.finish()


if __name__ == "__main__":
    build()
    print(OUT)

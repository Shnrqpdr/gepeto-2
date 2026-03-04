"""
Scraper de artigos do Wikipedia para corpus de treino do Gepeto-2.

Foca em artigos de Matemática e Física. Usa a Wikipedia API para
extrair texto limpo (sem HTML), preservando estrutura de parágrafos.

Saída:
  data/corpus_wikipedia.txt    — corpus pronto para treino
  data/scraping/.progress.json — progresso para retomar depois

Uso:
  python data/scraping/wikipedia_scraper.py
  python data/scraping/wikipedia_scraper.py --max-per-category 500
  python data/scraping/wikipedia_scraper.py --output data/corpus.txt
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

CATEGORIES = [
    "Mathematics",
    "Physics",
    "Algebra",
    "Calculus",
    "Geometry",
    "Number_theory",
    "Topology",
    "Classical_mechanics",
    "Quantum_mechanics",
    "Thermodynamics",
    "Electromagnetism",
    "Theory_of_relativity",
    "Mathematical_analysis",
    "Linear_algebra",
]

API_URL = "https://en.wikipedia.org/w/api.php"
OUTPUT_FILE = Path("data/corpus_wikipedia.txt")
PROGRESS_FILE = Path("data/scraping/.progress.json")

MIN_CHARS = 300      # artigos menores que isso são descartados
DELAY = 1.0          # segundos entre requisições (respeitar Wikipedia)

HEADERS = {
    "User-Agent": "GepetoCrawler/1.0 (Educational GPT-2 project; gepeto-2)"
}

# Seções sem valor educacional (geralmente no final de artigos Wikipedia)
JUNK_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "bibliography", "footnotes", "citations",
}

# --------------------------------------------------------------------------- #
# Limpeza de texto
# --------------------------------------------------------------------------- #

def _remove_junk_sections(text: str) -> str:
    """Remove seções sem valor educacional (referências, links externos, etc.)."""
    lines = text.splitlines()
    result = []
    in_junk = False
    for line in lines:
        m = re.match(r'^==+\s*(.+?)\s*==+$', line)
        if m:
            in_junk = m.group(1).strip().lower() in JUNK_SECTIONS
        if not in_junk:
            result.append(line)
    return "\n".join(result)


def clean_text(text: str) -> str:
    """Limpa artefatos do texto extraído da Wikipedia API."""
    text = _remove_junk_sections(text)
    # Remove referências numéricas [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Remove espaços em branco no fim de cada linha
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Colapsa mais de 2 quebras de linha consecutivas em 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Wikipedia API
# --------------------------------------------------------------------------- #

def get_category_members(category: str, max_articles: int) -> list[str]:
    """Retorna títulos de artigos (não subcategorias) de uma categoria."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "page",
        "cmlimit": "500",
        "format": "json",
    }

    while len(titles) < max_articles:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for member in data.get("query", {}).get("categorymembers", []):
            titles.append(member["title"])

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont or len(titles) >= max_articles:
            break
        params["cmcontinue"] = cont
        time.sleep(DELAY)

    return titles[:max_articles]


def get_article_text(title: str) -> str | None:
    """Busca o texto completo de um artigo em plaintext via Wikipedia API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("ns") == 0 and "extract" in page:
            return page["extract"]
    return None


def _fetch_with_retry(fn, *args, retries: int = 3, **kwargs):
    """Executa `fn` com até `retries` tentativas em caso de erro de rede."""
    log = logging.getLogger(__name__)
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = DELAY * (2 ** attempt)
            log.warning(f"Erro (tentativa {attempt + 1}/{retries}): {e}. Aguardando {wait:.0f}s...")
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# Progresso (permite retomar de onde parou)
# --------------------------------------------------------------------------- #

def _load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return set(data.get("scraped", []))
    return set()


def _save_progress(scraped: set[str]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps({"scraped": sorted(scraped)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Formatação do corpus
# --------------------------------------------------------------------------- #

def format_article(title: str, text: str) -> str:
    """Formata um artigo para o corpus: cabeçalho + corpo + linha em branco final."""
    return f"# {title}\n\n{text}\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraper Wikipedia (Matemática/Física) para corpus do Gepeto-2"
    )
    parser.add_argument(
        "--max-per-category", type=int, default=200,
        help="Máximo de artigos por categoria (padrão: 200)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"Arquivo de saída (padrão: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    scraped = _load_progress()
    log.info(f"Iniciando. {len(scraped)} artigos já coletados anteriormente.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(args.output, "a", encoding="utf-8") as out:
        for category in CATEGORIES:
            log.info(f"── Categoria: {category}")

            try:
                titles = _fetch_with_retry(
                    get_category_members, category, args.max_per_category
                )
            except requests.RequestException as e:
                log.error(f"Falha ao buscar categoria '{category}': {e}")
                continue

            log.info(f"   {len(titles)} artigos encontrados")

            for title in titles:
                if title in scraped:
                    continue

                try:
                    text = _fetch_with_retry(get_article_text, title)
                except requests.RequestException as e:
                    log.warning(f"   Falha em '{title}': {e}. Pulando.")
                    scraped.add(title)
                    _save_progress(scraped)
                    continue

                if not text or len(text) < MIN_CHARS:
                    scraped.add(title)
                    _save_progress(scraped)
                    continue

                text = clean_text(text)
                if len(text) < MIN_CHARS:
                    scraped.add(title)
                    _save_progress(scraped)
                    continue

                out.write(format_article(title, text) + "\n")
                out.flush()

                scraped.add(title)
                _save_progress(scraped)

                total += 1
                log.info(f"   [{total:>4}] {title} ({len(text):,} chars)")
                time.sleep(DELAY)

    log.info(f"\nFinalizado! {total} artigos escritos em '{args.output}'.")


if __name__ == "__main__":
    main()

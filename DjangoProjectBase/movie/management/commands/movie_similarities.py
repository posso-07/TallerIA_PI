import os
import numpy as np

from pathlib import Path

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from openai import OpenAI

from movie.models import Movie


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_MOVIE_1 = "La captura"
DEFAULT_MOVIE_2 = "Castillo medieval"
DEFAULT_PROMPT  = "película sobre la Segunda Guerra Mundial"


# ---------------------------------------------------------------------------
# Utilidades — entorno
# ---------------------------------------------------------------------------

def safe_load_dotenv() -> str | None:
    """
    Carga openAI.env desde la raíz del proyecto o directorio padre.
    Retorna la ruta usada, o None si se usó el valor por defecto.
    """
    candidates = [
        "key2_1.env",
        "../key2_1.env",
        str(Path(__file__).resolve().parents[3] / "key2_1.env"),
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            load_dotenv(candidate)
            return candidate

    load_dotenv()  # intenta con valor por defecto
    return None


# ---------------------------------------------------------------------------
# Utilidades — embeddings
# ---------------------------------------------------------------------------

def get_embedding(client: OpenAI, text: str) -> np.ndarray:
    """Obtiene el embedding de un texto usando OpenAI."""
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    if a.size == 0 or b.size == 0:
        raise ValueError("Los embeddings no pueden estar vacíos.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Comando
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = "Compara la similitud entre películas usando embeddings de OpenAI (similitud coseno)"

    def add_arguments(self, parser):
        parser.add_argument("--movie1",  type=str, default=DEFAULT_MOVIE_1)
        parser.add_argument("--movie2",  type=str, default=DEFAULT_MOVIE_2)
        parser.add_argument("--prompt",  type=str, default=DEFAULT_PROMPT)

    def handle(self, *args, **options):
        # Cargar variables de entorno
        dotenv_path = safe_load_dotenv()
        self.stdout.write(
            self.style.NOTICE(f"Variables de entorno cargadas desde: {dotenv_path or '(auto)'}")
        )

        # Validar API key
        api_key = os.environ.get("openai_apikey")
        if not api_key:
            self.stderr.write(
                self.style.ERROR("openai_apikey no encontrada. Revisa openAI.env")
            )
            return

        client = OpenAI(api_key=api_key)

        # Obtener películas
        movie1 = self._get_movie(options["movie1"])
        movie2 = self._get_movie(options["movie2"])

        if not movie1 or not movie2:
            return

        # Generar embeddings de las películas
        self.stdout.write(
            self.style.NOTICE(f"Obteniendo embeddings de '{movie1.title}' y '{movie2.title}'...")
        )
        emb1 = get_embedding(client, movie1.description or "")
        emb2 = get_embedding(client, movie2.description or "")

        # Similitud entre películas
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(
            f"🎬 Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}"
        )

        # Similitud contra el prompt
        prompt_text = options["prompt"]
        self.stdout.write(
            self.style.NOTICE(f"Obteniendo embedding del prompt: '{prompt_text}'...")
        )
        prompt_emb = get_embedding(client, prompt_text)

        sim_vs_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_vs_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"📝 Similitud prompt vs '{movie1.title}': {sim_vs_movie1:.4f}")
        self.stdout.write(f"📝 Similitud prompt vs '{movie2.title}': {sim_vs_movie2:.4f}")

    def _get_movie(self, title: str):
        """Busca una película por título y muestra error si no existe."""
        try:
            return Movie.objects.get(title=title)
        except Movie.DoesNotExist:
            self.stderr.write(
                self.style.ERROR(f"No existe película con título '{title}'")
            )
            return None
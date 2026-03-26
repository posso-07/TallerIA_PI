import os
import re

from django.conf import settings
from django.core.management.base import BaseCommand

from movie.models import Movie


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Normaliza texto para comparar nombres de archivos y títulos."""
    value = value or ""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def build_image_candidates(title: str) -> list:
    """Genera posibles claves normalizadas para buscar la imagen de una película."""
    normalized_title = normalize_text(title)

    if not normalized_title:
        return []

    return [
        normalized_title,
        f"m{normalized_title}",  # sin guion bajo porque normalize_text lo elimina
    ]


# ---------------------------------------------------------------------------
# Comando
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = "Asigna imágenes existentes en media/movie/images/ a las películas en la base de datos"

    def handle(self, *args, **kwargs):
        images_folder = os.path.join(settings.MEDIA_ROOT, "movie", "images")

        # Verificar que la carpeta exista
        if not os.path.isdir(images_folder):
            self.stderr.write(
                self.style.ERROR(f"No se encontró el directorio de imágenes: {images_folder}")
            )
            return

        self.stdout.write(self.style.WARNING(f"Usando carpeta de imágenes: {images_folder}"))

        # Construir índice de imágenes disponibles
        image_index = self._build_image_index(images_folder)

        if not image_index:
            self.stderr.write(
                self.style.ERROR("No se encontraron archivos de imagen compatibles en la carpeta.")
            )
            return

        # Procesar películas
        self._assign_images(image_index)

    def _build_image_index(self, images_folder: str) -> dict:
        """Construye un índice {nombre_normalizado: [archivos]} de las imágenes disponibles."""
        image_index = {}
        allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

        for fname in os.listdir(images_folder):
            _, ext = os.path.splitext(fname)

            if ext.lower() not in allowed_ext:
                continue

            base_name = os.path.splitext(fname)[0]
            normalized = normalize_text(base_name)

            if normalized:
                image_index.setdefault(normalized, []).append(fname)

            # Si el archivo empieza por m_, también indexarlo sin ese prefijo
            if base_name.lower().startswith("m_"):
                without_prefix = normalize_text(base_name[2:])
                if without_prefix:
                    image_index.setdefault(without_prefix, []).append(fname)

        return image_index

    def _assign_images(self, image_index: dict) -> None:
        """Recorre las películas y asigna la imagen correspondiente si existe."""
        movies = Movie.objects.all()
        total = movies.count()
        updated = 0
        no_match = 0

        self.stdout.write(self.style.SUCCESS(f"Encontradas {total} películas en la base de datos."))

        for movie in movies:
            candidates = build_image_candidates(movie.title)
            chosen_file = None

            # Buscar la primera coincidencia entre los candidatos
            for candidate in candidates:
                if candidate in image_index:
                    chosen_file = image_index[candidate][0]
                    break

            if chosen_file:
                relative_image_path = os.path.join("movie", "images", chosen_file).replace("\\", "/")
                movie.image = relative_image_path
                movie.save(update_fields=["image"])
                updated += 1
                self.stdout.write(self.style.SUCCESS(f"  ✔ {movie.title}: asignada {relative_image_path}"))
            else:
                no_match += 1
                self.stdout.write(self.style.WARNING(f"  ✘ {movie.title}: no se encontró imagen."))

        self.stdout.write(
            self.style.SUCCESS(f"\nFinalizado: {updated} actualizadas, {no_match} sin imagen.")
        )
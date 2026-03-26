import os
import random
import numpy as np
from dotenv import load_dotenv

from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI

def cosine_similarity(a, b):
    """Calcula la similitud coseno entre dos vectores."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

class Command(BaseCommand):
    help = 'Selecciona una película al azar y gestiona su embedding (DB o OpenAI).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--use-db-emb', 
            action='store_true', 
            help='Usar el embedding guardado en la BD si existe'
        )

    def handle(self, *args, **options):
        # 1. Configuración de Entorno
        load_dotenv('../key2_1.env')
        api_key = os.getenv('openai_apikey')
        
        if not api_key:
            self.stderr.write(self.style.ERROR('Error: openai_apikey no encontrada.'))
            return

        client = OpenAI(api_key=api_key)

        # 2. Selección de Datos
        movies = Movie.objects.all()
        if not movies.exists():
            self.stderr.write(self.style.ERROR('No hay películas en la base de datos.'))
            return

        movie = random.choice(list(movies))
        self.stdout.write(self.style.NOTICE(f"\nPelícula: {movie.title}"))
        self.stdout.write(f"Descripción: {movie.description[:100]}...")

        # 3. Obtención del Embedding
        embedding = None

        # Intentar cargar desde BD
        if options['use_db_emb'] and movie.emb:
            try:
                embedding = np.frombuffer(movie.emb, dtype=np.float32)
                self.stdout.write(self.style.SUCCESS('✓ Embedding cargado desde la BD.'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error al leer buffer: {e}"))

        # Generar con OpenAI si no existe o no se pidió de la BD
        if embedding is None:
            self.stdout.write('Generando embedding con OpenAI...')
            try:
                response = client.embeddings.create(
                    input=[movie.description], 
                    model='text-embedding-3-small'
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)

                # Guardar si el campo estaba vacío
                if not movie.emb:
                    movie.emb = embedding.tobytes()
                    movie.save(update_fields=['emb'])
                    self.stdout.write(self.style.SUCCESS('✓ Nuevo embedding guardado en BD.'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error en la API de OpenAI: {e}"))
                return

        # 4. Mostrar Resultados
        self.stdout.write(f"Longitud del vector: {len(embedding)}")
        self.stdout.write(f"Primeros 5 valores: {embedding[:5]}")

        # 5. Prueba de Similitud
        self.calculate_test_similarity(client, embedding)

    def calculate_test_similarity(self, client, movie_embedding):
        """Genera un prompt de prueba y compara la similitud."""
        prompt = 'película sobre la Segunda Guerra Mundial'
        try:
            res = client.embeddings.create(input=[prompt], model='text-embedding-3-small')
            prompt_emb = np.array(res.data[0].embedding, dtype=np.float32)
            
            score = cosine_similarity(prompt_emb, movie_embedding)
            
            self.stdout.write("---")
            self.stdout.write(self.style.NOTICE(f"Similitud con '{prompt}': {score:.4f}"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"No se pudo calcular la similitud: {e}"))
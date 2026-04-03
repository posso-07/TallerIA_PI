from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie

import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Cargar automáticamente variables de entorno desde archivos comunes
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / 'key2_1.env')
load_dotenv(BASE_DIR.parent / 'openAI.env')

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
import json


def get_ai_movie_suggestion(prompt, movies_list):
    """
    Recibe un prompt y una lista de películas de la BD.
    La IA elige cuál película de la lista recomienda.
    """
    api_key = os.getenv('openai_apikey')
    if not api_key:
        return None, 'No existe una API Key de OpenAI en la variable de entorno openai_apikey.'

    # Construir el listado de películas para enviarle a la IA
    movies_text = "\n".join(
        [f"- {m.title} ({m.year or 'sin año'}) | Género: {m.genre or 'desconocido'} | Descripción: {m.description[:100]}..."
         for m in movies_list]
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "user",
                    "content": f"""
Eres un asistente que recomienda películas. Solo puedes recomendar películas de la siguiente lista, no puedes inventar ni sugerir películas que no estén en ella.

Lista de películas disponibles:
{movies_text}

Basándote en el siguiente prompt del usuario, elige UNA película de la lista anterior que mejor se ajuste.
Devuelve ÚNICAMENTE un JSON válido con estas llaves: movie_title, reason.
Ejemplo: {{"movie_title": "La lista de Schindler", "reason": "Es una película sobre la Segunda Guerra Mundial."}}

Prompt del usuario: {prompt}
"""
                }
            ],
            temperature=0.7,
            max_tokens=300,
        )

        ai_text = response.choices[0].message.content.strip()

        if not ai_text:
            return None, 'La IA no devolvió contenido en la respuesta.'

        # Limpiar posibles bloques de código markdown
        ai_text = ai_text.replace('```json', '').replace('```', '').strip()

        try:
            parsed = json.loads(ai_text)
            movie_title = parsed.get('movie_title')
            reason = parsed.get('reason', '')
            if not movie_title:
                return None, 'La IA no pudo elegir una película.'
            return {'title': movie_title, 'reason': reason}, None
        except Exception:
            return None, 'La IA devolvió una respuesta con formato inválido.'

    except Exception as e:
        return None, f'Error al llamar a OpenAI: {e}'


def ai_recommendations(request):
    prompt = request.GET.get('prompt', '').strip()
    recommendation = None
    error = None

    if prompt:
        # Obtener todas las películas de la base de datos
        all_movies = list(Movie.objects.all())

        if not all_movies:
            error = 'No hay películas en la base de datos.'
        else:
            ai_result, ai_error = get_ai_movie_suggestion(prompt, all_movies)

            if ai_error:
                error = ai_error
            else:
                chosen_title = ai_result['title']
                reason = ai_result['reason']

                # Buscar la película en la BD (búsqueda flexible)
                movie = Movie.objects.filter(title__icontains=chosen_title).first()

                if not movie:
                    # Intentar buscar por palabras clave del título
                    for word in chosen_title.split():
                        if len(word) > 3:
                            movie = Movie.objects.filter(title__icontains=word).first()
                            if movie:
                                break

                if movie:
                    recommendation = {
                        'title': movie.title,
                        'description': movie.description,
                        'image': movie.image,
                        'genre': movie.genre,
                        'year': movie.year,
                        'reason': reason,
                    }
                else:
                    error = f'La IA recomendó "{chosen_title}" pero no se encontró en la base de datos. Intenta otro prompt.'

    return render(request, 'ai_recommendations.html', {
        'prompt': prompt,
        'recommendation': recommendation,
        'error': error,
    })


def home(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})


def about(request):
    return render(request, 'about.html')


def signup(request):
    email = request.GET.get('email')
    return render(request, 'signup.html', {'email': email})


def statistics_view0(request):
    matplotlib.use('Agg')
    all_movies = Movie.objects.all()

    movie_counts_by_year = {}
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    bar_width = 0.5
    bar_positions = range(len(movie_counts_by_year))

    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)
    plt.subplots_adjust(bottom=0.3)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'statistics.html', {'graphic': graphic})


def statistics_view(request):
    matplotlib.use('Agg')
    all_movies = Movie.objects.all()

    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    movie_counts_by_genre = {}
    for movie in all_movies:
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic
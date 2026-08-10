import os

import requests
from minsearch import Index
from openai import OpenAI


GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'


def build_gemini_client(api_key_env='GEMINI_API_KEY', base_url=GEMINI_BASE_URL):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f'{api_key_env} is not set in your environment.')

    return OpenAI(api_key=api_key, base_url=base_url)


def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents


def build_index(documents):
    index = Index(
        text_fields=['question', 'section', 'answer'],
        keyword_fields=['course']
    )
    index.fit(documents)
    return index

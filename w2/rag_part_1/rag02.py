import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

books = [
  {
    "title": "Introduction to Probability",
    "content": ("Joseph K. Blitzstein and Jessica Hwang's textbook (also known as 'the Harvard stats 110 book') "
                "provides an accessible introduction to probability theory with engaging examples and exercises.")
  },
  {
    "title": "A First Course in Probability",
    "content": ("Sheldon Ross's widely-used textbook covers fundamental probability concepts, random variables, "
                "expectation, and stochastic processes with mathematical rigor and practical examples.")
  },
  {
    "title": "Linear Algebra and Its Applications",
    "content": ("Gilbert Strang's classic textbook combines theoretical foundations with practical applications, "
                "making it one of the most popular introductions to linear algebra for engineers and scientists.")
  },
  {
    "title": "Linear Algebra Done Right",
    "content": ("Sheldon Axler's approach to linear algebra emphasizes conceptual understanding by avoiding "
                "determinants and focusing on vector spaces and linear transformations.")
  },
  {
    "title": "Pattern Recognition and Machine Learning",
    "content": ("Christopher Bishop's landmark textbook presents a comprehensive introduction to machine "
                "learning, combining mathematical depth with practical applications and algorithms.")
  },
  {
    "title": "Designing Machine Learning Systems",
    "content": ("Chip Huyen's practical guide explores the challenges of building machine learning systems "
                "that can be deployed reliably in production environments.")
  },
  {
    "title": "The LLM Engineer's Handbook",
    "content": ("This comprehensive guide covers the practical aspects of working with and deploying "
                "large language models, including prompt engineering, fine-tuning, and evaluation methods.")
  },
  {
    "title": "Illustrating Large Language Models",
    "content": ("Jay Alammar's visual guide breaks down complex concepts in LLMs through intuitive "
                "illustrations, making transformer architectures and attention mechanisms accessible.")
  },
  {
    "title": "System Design Interview - An Insider's Guide",
    "content": ("Alex Xu's guide helps software engineers prepare for system design interviews by walking "
                "through common design scenarios and providing detailed approaches to solving them.")
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [f'{book["title"]}. {book["content"]}' for book in books]

embeddings = model.encode(texts)

book_id = 0
similarities = cosine_similarity([embeddings[book_id]], embeddings)[0]
similar_indices = np.argsort(similarities)[::-1][1:3]
recommendations = [books[idx] for idx in similar_indices]
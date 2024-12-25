# RAG for Resume Search and Summarization

This repository contains the code and resources for the research project titled **Retrieval Augmented Generation (RAG) for Resume Search and Summarization**. This project demonstrates a cutting-edge solution for improving resume retrieval and summarization using advanced natural language processing techniques, blending semantic search with large language models.

The research paper can be found at: [Research Paper](https://drive.google.com/file/d/1WHLwXt19eklPeDi5c8rfNA2M0Mkuirz_/view)

## Overview

In today's recruitment landscape, the volume of job applications poses challenges for accurate and efficient resume screening. This project employs **Retrieval Augmented Generation (RAG)** to address these challenges by:

- Utilizing vector embeddings to semantically match resumes with user queries.
- Leveraging state-of-the-art language models (GPT-3.5, GPT-4, and Claude) for summarization.
- Generating concise, contextually relevant summaries that streamline decision-making for recruiters.

## Features

- **Semantic Search**: Employs OpenAI's `text-embedding-3-large` model to convert resumes into high-dimensional vectors stored in a **Chroma DB** vector database.
- **Dynamic Summarization**: Generates structured summaries highlighting skills, experiences, and qualifications tailored to user queries.
- **Multi-Model Comparison**: Evaluates multiple language models (GPT and Claude) for retrieval accuracy and summarization quality.

## Methodology

### Dataset
- **Size**: 3400+ resumes across diverse job categories (e.g., IT, HR, Design).
- **Structure**: Resumes stored in string and HTML formats, labeled by job categories.

### Pipeline
1. **Vectorization**: Resumes converted to vectors using OpenAI embeddings.
2. **Query Processing**: User queries mapped to vectors for cosine similarity search.
3. **Retrieval**: Top N matching resumes fetched from the vector database.
4. **Summarization**: Language models generate structured summaries based on retrieved resumes.

### Models Used
- **OpenAI**: GPT-3.5 Turbo and GPT-4 Turbo
- **Anthropic**: Claude-3 Series (Haiku, Sonnet, Opus)

### Evaluation
- Conducted with 20 diverse test cases using LangChain's evaluation framework.
- Metrics: Relevance, accuracy, format compliance, clarity, and coherence.

## Results

- **Best Performers**: GPT-4 Turbo and Claude-3 Opus outperformed others with superior accuracy and structured output.
- **Efficiency**: Significant improvement over traditional keyword-based methods in both retrieval precision and summarization quality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from supabase import create_client
from google import genai
from google.genai import types
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Labour Laws RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Initialize API clients
# -----------------------------------

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

groq = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------
# Hardcoded test query
# -----------------------------------

class QuestionRequest(BaseModel):
    question: str


# -----------------------------------
# STEP 1: Generate query embedding
# -----------------------------------

def generate_embedding(text):

    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=1536
        )
    )

    return response.embeddings[0].values


# -----------------------------------
# STEP 2: Supabase similarity search
# -----------------------------------

def retrieve_chunks(query_embedding):

    response = supabase.rpc(
        "match_labour_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    return response.data


# -----------------------------------
# STEP 3: Generate answer using Groq
# -----------------------------------

def generate_answer(query, chunks):

    context = "\n\n".join(
        [
            f"""
Act: {chunk['act_name']}
Year: {chunk['act_year']}
Section: {chunk['section_number']}
Jurisdiction: {chunk['jurisdiction']}

Content:
{chunk['content']}
"""
            for chunk in chunks
        ]
    )

    completion = groq.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": """
You are an AI legal assistant specializing in Indian labour laws. Your goal is to help ordinary people understand their legal rights using clear, simple and natural language.

Answer the user's question using the provided context.

INSTRUCTIONS:

1. Start with a direct, simple explanation that answers the user's question. Write as if you are explaining the law to someone who has no legal background.

2. Naturally introduce relevant legal details after the explanation. Mention the applicable Act, Section numbers and their significance. Explain legal terminology in simple words wherever necessary.

3. Keep the response conversational and well-organized. Use short paragraphs and bullet points wherever they genuinely improve readability.

4. Do NOT create rigid sections with headings such as "Simple Explanation", "Legal Details", or "Important Notes". The answer should flow naturally, as if a knowledgeable person is explaining something.

5. Do NOT create tables unless the user explicitly asks for one. Prefer short paragraphs and bullet points.

6. Avoid unnecessary technical jargon, lengthy legal quotations and repetitive explanations.

7. Preserve important legal terminology, Section numbers and Act names. Explain what they mean rather than removing them.

8. Never sacrifice legal accuracy for simplicity. Do not invent laws, sections, penalties or interpretations.

9. Use only the provided context for legal claims. If the retrieved context does not contain enough information, clearly communicate the limitation rather than guessing.

10. Distinguish between what the law explicitly states and your simplified explanation of its practical meaning.

The overall tone should be friendly, clear, trustworthy and informative. The user should feel that they are talking to a knowledgeable assistant, not reading a legal textbook.
"""
            },
            {
                "role": "user",
                "content": f"""
Question:
{query}

Retrieved Legal Context:
{context}

Provide a clear and accurate answer.
"""
            }
        ],

        temperature=0.2
    )

    return completion.choices[0].message.content


# -----------------------------------
# STEP 4: FastAPI endpoint
# -----------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    query = request.question.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)

        # Retrieve relevant chunks
        chunks = retrieve_chunks(query_embedding)

        if not chunks:
            return {
                "query": query,
                "message": "No relevant chunks found.",
                "retrieved_chunks": 0
            }

        # Generate answer
        answer = generate_answer(query, chunks)

        return {
            "query": query,
            "retrieved_chunks": len(chunks),
            "answer": answer,
            "sources": [
                {
                    "act_name": c["act_name"],
                    "act_year": c["act_year"],
                    "section_number": c["section_number"],
                    "jurisdiction": c["jurisdiction"]
                }
                for c in chunks
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
def health_check():
    return {"status": "ok"}

"""Central configuration for LangChain model objects.

Developers should configure API keys, model names, and base URLs here.
"""

from lamb.rate_limited_chat import RateLimitedChatOpenAI, RateLimitedOpenAIEmbeddings


chat_model = RateLimitedChatOpenAI(
    model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    base_url = "http://localhost:7217/v1",
    api_key = "YOUR_API_KEY",
    temperature=0.0,
)

# Configure Embeddings Model for Dense Vector Retrieval
embeddings_model = RateLimitedOpenAIEmbeddings(
    model = "Qwen/Qwen3-Embedding-8B",
    base_url = "http://localhost:7218/v1",
    api_key = "YOUR_API_KEY",
)

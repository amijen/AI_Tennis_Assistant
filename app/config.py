from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # LLM 
    groq_api_key : str
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings 
    embedding_model : str = "BAAI/bge-base-en-v1.5"

    # database
    database_url : str

    # App
    app_env : str = "development"

    @property 
    def is_dev(self) -> bool : 
        return self.app_env == "development"

    class Config: 
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", case_sensitive=False, extra="ignore")

    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: str | None = Field(default=None, description="Qdrant API key (optional)")
    default_collection_name: str = Field(
        default="repo_documentation_embeddings",
        description="Default Qdrant collection name for storing document embeddings",
    )
    # We can add other Qdrant specific settings here if needed, e.g., timeout, https


qdrant_settings = QdrantSettings()

# Example of how to load other settings if needed in the core module
# class CoreSettings(BaseSettings):
#     some_other_setting: str = "default_value"

# core_settings = CoreSettings()

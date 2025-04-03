from pydantic_settings import BaseSettings
# import the base class for creating settings models

class Settings(BaseSettings):
    # BaseSettings is a class that helps manage application settings.
    # It loads environment variables and provides a convenient way to access them.
    # create a subclass called Settings that inherits from BaseSettings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # optional
    APP_ENV: str = "development" # optional, tracks environment (dev/staging/prod)
    DEBUG: bool = False # optional, tracks if the app is in debug mode

    class Config:
        # Config is a nested class that provides configuration options for the Settings class.
        # It is a speical class defined by Pydantic that allows you to customize the behavior of the model.
        env_file = ".env"

settings = Settings()
# instantiate the settings
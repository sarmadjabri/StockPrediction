import logging

def setup_logging():
    """Set up logging configuration."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

if __name__ == "__main__":
    setup_logging()

import logging
from src.config import Config
from src.train import train
from src.evaluate import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = Config("configs/config.yaml")

    logger.info("Starting Multimodal Healthcare AI application")

    logger.info("Training model...")
    train(config)

    logger.info("Evaluating model...")
    evaluate(config)

    logger.info("Application completed successfully")

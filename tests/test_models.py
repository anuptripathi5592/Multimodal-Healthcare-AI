import unittest
from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder

class TestModels(unittest.TestCase):
    def test_vision_encoder(self):
        encoder = VisionEncoder(output_dim=512)
        self.assertIsNotNone(encoder)
    
    def test_text_encoder(self):
        encoder = TextEncoder(output_dim=512)
        self.assertIsNotNone(encoder)

if __name__ == '__main__':
    unittest.main()

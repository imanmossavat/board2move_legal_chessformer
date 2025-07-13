import unittest
import torch
from move_vocab_builder import load_or_build_vocab

from model import MinimalChessTransformer  # replace with your actual model file name


class TestMinimalChessTransformer(unittest.TestCase):

    def setUp(self):
        # Build vocab and get number of classes
        self.uci_to_index, index_to_uci, from_ids, to_ids, promo_ids = load_or_build_vocab()

        self.num_classes = len(self.uci_to_index)
        self.input_dim = 13
        self.batch_size = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate model
        self.model = MinimalChessTransformer(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            device=self.device
        ).to(self.device)

        self.model.eval()

    def test_forward_pass_output_shape(self):
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 64, self.input_dim).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(dummy_input)

        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape, f"Expected output shape {expected_shape} but got {output.shape}")


if __name__ == "__main__":
    unittest.main()

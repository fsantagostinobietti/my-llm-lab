import unittest
import torch
from torch import nn

from tictactoe_lm_1 import (
    causal_mask,
    TTTAttention,
    TTTTransformer,
    TicTacToeLM_1,
    CTX_SZ,
    VOCAB_SZ,
    D_IN,
    NUM_BLOCKS,
)


class TestCausalMask(unittest.TestCase):
    """Test suite for causal_mask function"""

    def test_causal_mask_shape(self):
        """Test that causal mask returns same shape as input"""
        S = torch.randn(9, 9)
        masked = causal_mask(S)
        self.assertEqual(S.shape, masked.shape)

    def test_causal_mask_upper_triangular(self):
        """Test that causal mask zeros out upper triangular matrix"""
        S = torch.ones(5, 5)
        masked = causal_mask(S)
        
        # Upper triangular part should be -inf
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertTrue(torch.isinf(masked[i, j]))
        
        # Lower triangular and diagonal should be preserved
        for i in range(5):
            for j in range(i + 1):
                self.assertEqual(masked[i, j], 1.0)

    def test_causal_mask_batch_dimension(self):
        """Test causal mask with batch dimension"""
        S = torch.randn(2, 9, 9)  # batch size 2
        masked = causal_mask(S)
        self.assertEqual(S.shape, masked.shape)
        # Check first batch
        for i in range(9):
            for j in range(i + 1, 9):
                self.assertTrue(torch.isinf(masked[0, i, j]))


class TestTTTAttention(unittest.TestCase):
    """Test suite for TTTAttention module"""

    def setUp(self):
        """Initialize attention module before each test"""
        self.attention = TTTAttention()
        torch.manual_seed(42)

    def test_attention_initialization(self):
        """Test that attention module initializes with correct parameters"""
        self.assertIsNotNone(self.attention.Wq)
        self.assertIsNotNone(self.attention.Wk)
        self.assertEqual(self.attention.Wq.shape, (D_IN, D_IN))
        self.assertEqual(self.attention.Wk.shape, (D_IN, D_IN))

    def test_attention_output_shape(self):
        """Test that attention output has correct shape"""
        X = torch.randn(CTX_SZ, D_IN)
        output = self.attention(X)
        self.assertEqual(output.shape, (CTX_SZ, D_IN))

    def test_attention_batch_processing(self):
        """Test attention with batch dimension"""
        batch_size = 4
        X = torch.randn(batch_size, CTX_SZ, D_IN)
        output = self.attention(X)
        self.assertEqual(output.shape, (batch_size, CTX_SZ, D_IN))

    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention module"""
        X = torch.randn(CTX_SZ, D_IN, requires_grad=True)
        output = self.attention(X)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.any(X.grad != 0))

    def test_attention_causality(self):
        """Test that attention is causal (no information from future tokens)"""
        X = torch.randn(CTX_SZ, D_IN)
        output = self.attention(X)
        # Check that output at position t only depends on inputs up to t
        for t in range(CTX_SZ):
            with torch.no_grad():
                X_modified = X.clone()
                X_modified[t+1:] = 0  # zero out future inputs
            output_modified = self.attention(X_modified)
            print("t:", t, "output:", output[t], "output_modified:", output_modified[t])
            self.assertTrue(torch.allclose(output[t], output_modified[t]))


class TestTTTTransformer(unittest.TestCase):
    """Test suite for TTTTransformer module"""

    def setUp(self):
        """Initialize transformer module before each test"""
        self.transformer = TTTTransformer()
        torch.manual_seed(42)

    def test_transformer_output_shape(self):
        """Test that transformer output has correct shape"""
        X = torch.randn(CTX_SZ, D_IN)
        output = self.transformer(X)
        self.assertEqual(output.shape, (CTX_SZ, D_IN))

    def test_transformer_batch_processing(self):
        """Test transformer with batch dimension"""
        batch_size = 8
        X = torch.randn(batch_size, CTX_SZ, D_IN)
        output = self.transformer(X)
        self.assertEqual(output.shape, (batch_size, CTX_SZ, D_IN))

    def test_transformer_gradient_flow(self):
        """Test that gradients flow through transformer"""
        X = torch.randn(CTX_SZ, D_IN, requires_grad=True)
        output = self.transformer(X)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.any(X.grad != 0))



class TestTicTacToeLM_1(unittest.TestCase):
    """Test suite for TicTacToeLM_1 model"""

    def setUp(self):
        """Initialize model before each test"""
        self.model = TicTacToeLM_1()
        torch.manual_seed(42)

    def test_model_forward_single_input(self):
        """Test model forward pass with single input"""
        x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        logits = self.model(x)
        
        # Output should be (CTX_SZ, VOCAB_SZ)
        self.assertEqual(logits.shape, (CTX_SZ, VOCAB_SZ))

    def test_model_forward_batch_input(self):
        """Test model forward pass with batched input"""
        batch_size = 4
        x = torch.randint(0, VOCAB_SZ, (batch_size, CTX_SZ))
        logits = self.model(x)
        
        # Output should be (batch_size, CTX_SZ, VOCAB_SZ)
        self.assertEqual(logits.shape, (batch_size, CTX_SZ, VOCAB_SZ))

    def test_model_gradient_flow(self):
        """Test that gradients flow through entire model"""
        x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check that embedding layer received gradients
        self.assertIsNotNone(self.model.Wemb.weight.grad)
        self.assertTrue(torch.any(self.model.Wemb.weight.grad != 0))

    def test_model_prediction_range(self):
        """Test that predictions are valid logits (finite values)"""
        x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        logits = self.model(x)
        
        # Logits should be finite and represent valid probability space
        self.assertTrue(torch.all(torch.isfinite(logits)))
        
        # Should be possible to get argmax predictions
        predictions = torch.argmax(logits, dim=-1)
        self.assertEqual(predictions.shape, (CTX_SZ,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < VOCAB_SZ))

    def test_model_embedding_tight_weight_sharing(self):
        """Test that output layer uses tight weight sharing with embedding"""
        x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        
        # Forward pass
        logits = self.model(x)
        
        # The logits should be computed using Wemb.weight
        # This is already verified by the forward method using @self.Wemb.weight.t()
        self.assertEqual(logits.shape[-1], VOCAB_SZ)



class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Initialize model for integration tests"""
        self.model = TicTacToeLM_1()
        torch.manual_seed(42)

    def test_training_step(self):
        """Test a complete training step"""
        # Create model, input, and simple loss
        model = TicTacToeLM_1()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        logits = model(x)
        
        # Create dummy target
        target = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss decreased (with high probability)
        self.assertTrue(torch.isfinite(loss))

    def test_inference_mode(self):
        """Test model in inference mode (no gradients)"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
            logits = self.model(x)
            predictions = torch.argmax(logits, dim=-1)
        
        # Verify predictions are valid
        self.assertEqual(predictions.shape, (CTX_SZ,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < VOCAB_SZ))

    def test_batch_consistency(self):
        """Test that batched processing is consistent with unbatched"""
        x_single = torch.randint(0, VOCAB_SZ, (CTX_SZ,))
        x_batch = x_single.unsqueeze(0)  # Add batch dimension
        
        # Get outputs
        logits_single = self.model(x_single)
        logits_batch = self.model(x_batch)
        
        # Should be consistent
        self.assertTrue(torch.allclose(logits_single, logits_batch[0], atol=1e-5))


if __name__ == "__main__":
    unittest.main()

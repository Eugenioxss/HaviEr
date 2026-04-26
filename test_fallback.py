import os
import sys
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the environment variables before importing app
os.environ['GEMINI_API_KEY'] = 'test_key'
os.environ['MIMO_API_KEY'] = 'test_mimo_key'
os.environ['FALLBACK_PRIORITY'] = 'mimo,gemini'

# Import the functions to test
from app import llamar_mimo, llamar_gemini, llamar_con_fallback

def test_llamar_mimo_success():
    """Test llamar_mimo with successful response."""
    with patch('app.mimo_client') as mock_mimo_client:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test MiMo response"
        mock_mimo_client.chat.completions.create.return_value = mock_response
        
        text, error = llamar_mimo("Test prompt", temperatura=0.7)
        
        assert text == "Test MiMo response"
        assert error is None
        print("✅ test_llamar_mimo_success passed")

def test_llamar_mimo_failure():
    """Test llamar_mimo with failed response."""
    with patch('app.mimo_client') as mock_mimo_client:
        mock_mimo_client.chat.completions.create.side_effect = Exception("API Error")
        
        text, error = llamar_mimo("Test prompt", temperatura=0.7)
        
        assert text is None
        assert error is not None
        assert error[1] == 500
        print("✅ test_llamar_mimo_failure passed")

def test_llamar_con_fallback_mimo_success():
    """Test llamar_con_fallback with MiMo success."""
    with patch('app.mimo_client') as mock_mimo_client:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test MiMo response"
        mock_mimo_client.chat.completions.create.return_value = mock_response
        
        text, error = llamar_con_fallback("Test prompt", temperatura=0.7)
        
        assert text == "Test MiMo response"
        assert error is None
        print("✅ test_llamar_con_fallback_mimo_success passed")

def test_llamar_con_fallback_gemini_fallback():
    """Test llamar_con_fallback with MiMo failure and Gemini success."""
    with patch('app.mimo_client') as mock_mimo_client, \
         patch('app.llamar_gemini') as mock_llamar_gemini:
        
        # MiMo fails
        mock_mimo_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Gemini succeeds
        mock_llamar_gemini.return_value = ("Test Gemini response", None)
        
        text, error = llamar_con_fallback("Test prompt", temperatura=0.7)
        
        assert text == "Test Gemini response"
        assert error is None
        mock_llamar_gemini.assert_called_once()
        print("✅ test_llamar_con_fallback_gemini_fallback passed")

def test_llamar_con_fallback_both_fail():
    """Test llamar_con_fallback with both models failing."""
    with patch('app.mimo_client') as mock_mimo_client, \
         patch('app.llamar_gemini') as mock_llamar_gemini:
        
        # MiMo fails
        mock_mimo_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Gemini fails
        mock_llamar_gemini.return_value = (None, ("Gemini error", 500))
        
        text, error = llamar_con_fallback("Test prompt", temperatura=0.7)
        
        assert text is None
        assert error == ("Gemini error", 500)
        print("✅ test_llamar_con_fallback_both_fail passed")

if __name__ == "__main__":
    test_llamar_mimo_success()
    test_llamar_mimo_failure()
    test_llamar_con_fallback_mimo_success()
    test_llamar_con_fallback_gemini_fallback()
    test_llamar_con_fallback_both_fail()
    print("\n✅ All tests passed!")
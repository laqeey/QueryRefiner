# queryrefiner/models.py

"""Module for loading transformer models used by QueryRefiner."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline

def load_models(code_model, device):
    """Load CodeBERT, CodeT5p, and the specified code generation model.

    Args:
        code_model (str): The model name for code generation.
        device (int): The device ID for computation (-1 for CPU, 0+ for GPU).

    Returns:
        tuple: (tokenizer_bert, model_bert, tokenizer_t5, model_t5, generator)

    Raises:
        Exception: If model loading fails.
    """
    try:
        tokenizer_bert = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model_bert = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
        print("CodeBERT loaded successfully!")

        tokenizer_t5 = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", force_download=True)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m", force_download=True)
        print("CodeT5p loaded successfully!")

        generator = pipeline("text-generation", model=code_model, device=device)
        print(f"{code_model} loaded successfully!")

        return tokenizer_bert, model_bert, tokenizer_t5, model_t5, generator
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise
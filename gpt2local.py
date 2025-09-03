import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Local:
    """
    Clase para gestionar el modelo GPT-2 local.
    Permite generar texto, hacer fine-tuning y guardar/cargar el modelo.
    """

    def __init__(self, model_path="gpt2"):
        self.available = False
        self.model = None
        self.tokenizer = None
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"❌ Error cargando GPT-2: {e}")
            self.available = False

    def generate(self, prompt, max_length=100, temperature=0.8, top_p=0.9):
        if not self.available:
            return "Modelo GPT-2 local no disponible."
        try:
            structured_prompt = f"DODONEST AI Assistant created by Maikeiru.\nUser: {prompt}\nDODONEST:"
            inputs = self.tokenizer.encode(structured_prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + 50, max_length),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "DODONEST:" in generated_text:
                response = generated_text.split("DODONEST:")[-1].strip()
            else:
                response = generated_text[len(structured_prompt):].strip()
            return self.clean_response(response)
        except Exception as e:
            print(f"❌ Error generando con GPT-2 local: {e}")
            return "No puedo generar respuesta útil con el modelo local."

    def clean_response(self, response):
        if not response:
            return ""
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                if not any(line.lower() in prev.lower() for prev in cleaned_lines[-2:]):
                    cleaned_lines.append(line)
            if len(cleaned_lines) >= 3:
                break
        result = ' '.join(cleaned_lines)
        if len(result) > 200:
            result = result[:200] + "..."
        return result

    def fine_tune(self, train_texts, epochs=2, lr=5e-5):
        if not self.available:
            print("Modelo GPT-2 local no disponible para entrenamiento.")
            return
        from torch.optim import AdamW
        optimizer = AdamW(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for i, text in enumerate(train_texts):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    labels = inputs.input_ids.clone()
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print(f"Error en fine-tuning texto {i}: {e}")
                    continue
        self.model.eval()

    def save_model(self, path="./dodonest_model"):
        if not self.available:
            print("Modelo GPT-2 local no disponible para guardar.")
            return
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")

    def load_model(self, path="./dodonest_model"):
        try:
            if os.path.exists(path):
                self.model = GPT2LMHeadModel.from_pretrained(path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.eval()
                self.available = True
            else:
                print(f"⚠️ No se encontró modelo en {path}, usando modelo base")
        except Exception as e:
            print(f"❌ Error cargando modelo personalizado: {e}")
            self.available = False
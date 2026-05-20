"""
Asesor LLM local para orientación sobre lesiones cutáneas.
Modelo: TinyLlama-1.1B-Chat (~2.2GB en disco, ~2GB RAM/VRAM).
Alternativas ligeras: Qwen2-0.5B-Instruct (~1GB).

Uso:
  from llm_advisor import LLMAdvisor
  advisor = LLMAdvisor()
  advisor.load()
  respuesta = advisor.generate(diagnostico, confianza, descripcion, severidad)
"""

import os
import re
import textwrap

ADVISOR_AVAILABLE = False
SYSTEM_PROMPT = (
    "Eres un asistente de dermatología que proporciona información educativa "
    "sobre lesiones cutáneas. Tus respuestas deben ser claras, precisas y "
    "en español. Incluye SIEMPRE un descargo de responsabilidad: "
    "esta información no sustituye la consulta con un dermatólogo."
)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    ADVISOR_AVAILABLE = True
except ImportError:
    ADVISOR_AVAILABLE = False

MODEL_OPTIONS = {
    'TinyLlama-1.1B (recomendado)': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'Qwen2-0.5B (más ligero)': 'Qwen/Qwen2-0.5B-Instruct',
}


def _build_prompt(diagnosis, confidence, description, severity):
    diagnosis_clean = diagnosis.replace('_', ' ').title()
    return f"""\
{''.rjust(70, '=')}
INFORME DERMATOLÓGICO ASISTIDO POR IA
{''.rjust(70, '=')}
Diagnóstico: {diagnosis_clean}
Confianza:   {confidence:.1%}
Gravedad:    {severity}

Contexto clínico:
{description}

Basado en este diagnóstico, proporciona:
1. QUÉ ES: Explica brevemente esta condición cutánea
2. TRATAMIENTO: Opciones de tratamiento disponibles
3. CUÁNDO CONSULTAR: Signos que requieren atención médica
4. PRONÓSTICO: Expectativas generales

IMPORTANTE: Esta es información educativa. No es un diagnóstico médico profesional.
"""
def _build_prompt_qwen(diagnosis, confidence, description, severity):
    diagnosis_clean = diagnosis.replace('_', ' ').title()
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Diagnóstico del modelo: {diagnosis_clean}
Confianza: {confidence:.1%}
Gravedad: {severity}
Descripción: {description}

Proporciona información educativa sobre esta condición cutánea: qué es, opciones de tratamiento, cuándo consultar a un dermatólogo, y pronóstico general.<|im_end|>
<|im_start|>assistant
"""


def _build_prompt_tinyllama(diagnosis, confidence, description, severity):
    diagnosis_clean = diagnosis.replace('_', ' ').title()
    return f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Diagnóstico del modelo: {diagnosis_clean}
Confianza: {confidence:.1%}
Gravedad: {severity}
Descripción: {description}

Proporciona información educativa sobre esta condición cutánea: qué es, opciones de tratamiento, cuándo consultar a un dermatólogo, y pronóstico general.
<|assistant|>
"""


class LLMAdvisor:
    def __init__(self, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
        self.model_name = model_name
        self.pipe = None
        self.loaded = False
        self.device = None

    def load(self):
        if not ADVISOR_AVAILABLE:
            raise ImportError(
                "transformers no está instalado. Ejecuta: pip install transformers torch"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[LLM] Cargando {self.model_name} en {self.device}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            model_kwargs={"low_cpu_mem_usage": True} if self.device == "cpu" else {},
        )
        self.loaded = True
        print(f"[LLM] Modelo cargado correctamente en {self.device}")

    def generate(self, diagnosis, confidence, description, severity, max_new_tokens=512):
        if not self.loaded:
            raise RuntimeError("Modelo no cargado. Llama a .load() primero.")

        if 'tinyllama' in self.model_name.lower():
            prompt = _build_prompt_tinyllama(diagnosis, confidence, description, severity)
        else:
            prompt = _build_prompt_qwen(diagnosis, confidence, description, severity)

        result = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        raw = result[0]['generated_text']

        response = raw[len(prompt):].strip()
        response = re.sub(r'<\|im_end\|>|<\|assistant\|>|<\|user\|>', '', response)
        response = re.sub(r'\n{3,}', '\n\n', response).strip()
        return response

    def unload(self):
        self.pipe = None
        self.loaded = False
        if self.device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()


def format_response(text):
    lines = text.split('\n')
    formatted = []
    for line in lines:
        line = line.strip()
        if not line:
            formatted.append('')
        elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.'):
            formatted.append(f"\n**{line}**" if len(line) < 15 else f"\n**{line[:line.index(' ')]}** {line[line.index(' ')+1:]}")
        elif line[0].isdigit() and '. ' in line[:4]:
            idx = line.index(' ')
            formatted.append(f"\n**{line[:idx]}** {line[idx+1:]}")
        else:
            formatted.append(line)
    return '\n'.join(formatted)

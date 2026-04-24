from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_apertus_8b():
    model_id = 'swiss-ai/Apertus-8B-2509'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return HuggingfaceSubject(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        region_layer_mapping={
            # Here we take layer 12 because it has shown to be the best for Pereira
            # ArtificialSubject.RecordingTarget.language_system: ['model.layers.12']
            # We switched to layer 9 for lebel based on the experiments
            ArtificialSubject.RecordingTarget.language_system: 'model.layers.9'
            # report each layer for each benchmark and take the best performing layer
            # ArtificialSubject.RecordingTarget.language_system: [f'model.layers.{i}' for i in range(32)]
        },
    )


model_registry['apertus-8b'] = _load_apertus_8b

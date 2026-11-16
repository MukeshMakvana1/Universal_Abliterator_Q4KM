# Universal_Abliterator_Q4KM/core/domain_steerer.py
# ==============================================================================
# Non-Gradient Representation Steering & Identity Injection Engine
# Enhances coding/math reasoning capabilities and embeds custom identity
# ==============================================================================
import json
import time
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


# Canonical domain anchor terms for coding & algorithm specialization
CODING_ANCHOR_TERMS = [
    "def", "class", "return", "import", "async", "await", "yield",
    "lambda", "algorithm", "complexity", "optimal", "binary_search",
    "dynamic_programming", "recursion", "type", "List", "Dict", "Optional",
    "exception", "try", "except", "unittest", "assert", "self"
]


class DomainAndIdentitySteerer:
    """
    Steers model representations toward specific task domains (e.g. coding)
    and binds the custom persona/identity into the model's vocabulary and generation configs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        target_domain: str = "coding",
        steering_factor: float = 0.04,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_domain = target_domain
        self.steering_factor = float(steering_factor)
        self.device = model.device
        self.dtype = model.dtype

    def _discover_transformer_layers(self) -> Optional[nn.ModuleList]:
        """Dynamically locates the transformer layer stack across model architectures."""
        search_paths = [
            ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
            ("layers", lambda m: getattr(m, "layers", None)),
            ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
            ("model.transformer.h", lambda m: getattr(getattr(getattr(m, "model", None), "transformer", None), "h", None)),
            ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ]
        for name, getter in search_paths:
            try:
                candidate = getter(self.model)
                if candidate is not None and isinstance(candidate, (nn.ModuleList, list)) and len(candidate) > 0:
                    return candidate
            except Exception:
                continue

        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 4:
                return mod

        return None

    def steer_representations(self, layer_indices: Optional[List[int]] = None) -> int:
        """
        Calculates task vector for domain anchor tokens and steers middle-layer representations.
        """
        if self.target_domain != "coding":
            print(f"ℹ️ Steering mode '{self.target_domain}' - skipping domain steering.")
            return 0

        print(f"🚀 Constructing domain subspace steering vector for: '{self.target_domain}'...")
        t0 = time.time()

        # 1. Resolve token IDs for coding anchors
        token_ids = []
        for term in CODING_ANCHOR_TERMS:
            ids = self.tokenizer.encode(term, add_special_tokens=False)
            token_ids.extend(ids)
        token_ids = list(set(token_ids))

        # 2. Extract embedding representations
        embed_tokens = self.model.get_input_embeddings()
        if embed_tokens is None or not hasattr(embed_tokens, "weight"):
            print("⚠️ Input embedding layer not found. Skipping representation steering.")
            return 0

        embed_weights = embed_tokens.weight.data  # [vocab_size, hidden_size]
        domain_subspace = embed_weights[token_ids, :].to(dtype=torch.float32)

        # Compute centroid task vector
        domain_vector = torch.mean(domain_subspace, dim=0)  # [hidden_size]
        norm = torch.norm(domain_vector, p=2)
        if norm > 1e-6:
            domain_vector = domain_vector / norm
        domain_vector = domain_vector.to(device=self.device, dtype=self.dtype)

        # 3. Apply subspace steering to target layers
        layers = self._discover_transformer_layers()
        if layers is None:
            print("⚠️ Could not locate transformer layers for steering.")
            return 0

        total_layers = len(layers)
        if layer_indices is None:
            # Middle-to-late transformer layers (e.g. layers 10-22)
            layer_indices = list(range(int(total_layers * 0.4), int(total_layers * 0.9)))

        modified_count = 0
        mlp_candidates = [
            ("mlp", "down_proj"),
            ("mlp", "c_proj"),
            ("mlp", "fc2"),
            ("mlp", "dense_4h_to_h"),
            ("feed_forward", "w2"),
        ]

        with torch.no_grad():
            for l_idx in layer_indices:
                if l_idx >= total_layers:
                    continue
                block = layers[l_idx]

                # Find MLP down projection
                linear_layer = None
                for parent_attr, child_attr in mlp_candidates:
                    parent = getattr(block, parent_attr, None)
                    if parent is not None:
                        cand = getattr(parent, child_attr, None)
                        if cand is not None and isinstance(cand, nn.Linear):
                            linear_layer = cand
                            break

                if linear_layer is not None:
                    W = linear_layer.weight.data  # [d, k]
                    d, k = W.shape
                    if d != domain_vector.shape[0]:
                        continue
                    steer_mat = torch.outer(domain_vector, torch.ones(k, device=self.device, dtype=self.dtype) / k)
                    W.add_(self.steering_factor * steer_mat)
                    modified_count += 1

        elapsed = time.time() - t0
        print(f"✅ Steered {modified_count} layer matrices toward coding subspace in {elapsed:.2f}s!")
        return modified_count

    def inject_identity_configs(
        self,
        output_dir: Path,
        assistant_name: str,
        creator_name: str,
        system_prompt: str,
    ):
        """
        Updates generation_config.json, tokenizer_config.json, and chat_template.jinja
        to enforce the persona across any chat interface or IDE.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📝 Injecting identity '{assistant_name}' into model packaging...")

        # 1. Update generation_config.json
        gen_cfg_path = output_dir / "generation_config.json"
        gen_data = {}
        if gen_cfg_path.exists():
            try:
                with open(gen_cfg_path, "r", encoding="utf-8") as f:
                    gen_data = json.load(f)
            except Exception:
                pass

        gen_data.update({
            "assistant_name": assistant_name,
            "creator": creator_name,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "max_new_tokens": 1024,
            "do_sample": True,
        })
        with open(gen_cfg_path, "w", encoding="utf-8") as f:
            json.dump(gen_data, f, indent=2, ensure_ascii=False)

        # 2. Custom Chat Template (Jinja) embedding system prompt default
        chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\n" + system_prompt.replace("\n", " ").replace("'", "\\'") + "<|im_end|>\\n' }}"
            "{% endif %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

        template_path = output_dir / "chat_template.jinja"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(chat_template)

        # 3. Update tokenizer_config.json
        tok_cfg_path = output_dir / "tokenizer_config.json"
        if tok_cfg_path.exists():
            try:
                with open(tok_cfg_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                tok_data["chat_template"] = chat_template
                with open(tok_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(tok_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Could not update tokenizer_config.json: {e}")

        print("✅ Identity configs successfully injected.")

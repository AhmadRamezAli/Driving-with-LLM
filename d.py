import torch, json
from transformers import AutoTokenizer
from app.slices.prediction.ai.models.vector_lm import VectorLMWithLoRA     # your class
from app.slices.prediction.ai.utils.model_utils import load_model                # loads base + LoRA adapter
from app.slices.prediction.ai.utils.vector_utils.randomize_utils import VectorObservation
from app.slices.prediction.ai.utils.prompt_utils import make_observation_prompt
from app.slices.prediction.ai.action_parser import parse_actions
# ---------- 1.  load model & tokenizer ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(
    base_model      = "deepseek-ai/deepseek-coder-1.3b-base",
    resume_from_checkpoint = "last",   # <-- your fine-tuned weights
    lora_r = 4,
    lora_alpha = 8,
    lora_dropout = 0.05,
    lora_target_modules = ("q_proj","v_proj"),
    load_in_8bit    = False                             # fp16 or bfloat16 if you prefer
).eval().to(device)
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
SENTINEL  = torch.tensor([[10567, 29901]], device=device)    # “<VECTOR>”

# ---------- 2.  build textual prompt ----------
instruction = ("You are a certified professional driving instructor. "
               "Based only on the sensor inputs, first describe the scene, "
               "then recommend an action.")

header = (
    "### Instruction:\n"
    f"{instruction}\n"
    "### Input:\n"
                 # placeholder; we’ll replace by embeddings
    "### Response:\n"
)
enc = tokenizer(header, add_special_tokens=False, return_tensors="pt").to(device)
user_ids  = enc.input_ids
user_mask = enc.attention_mask

# ---------- 3.  prepare vector tensors ----------
with open("input_descriptors.json") as f: obs_dict = json.load(f)

def to_t(x): return torch.tensor(x, dtype=torch.float32, device=device)
obs = VectorObservation(**{k: to_t(v) for k, v in obs_dict.items()})
# Create a prompt string exactly as the model saw during training:
# raw_prompt = make_observation_prompt({
#     "route_descriptors":    obs.route_descriptors,
#     "vehicle_descriptors":  obs.vehicle_descriptors,
#     "pedestrian_descriptors": obs.pedestrian_descriptors,
#     "ego_vehicle_descriptor": obs.ego_vehicle_descriptor,
# })
# print(raw_prompt)
# print("--------------------------------")
# ---------- 4.  generation ----------



with torch.no_grad():
    gen_ids = model.generate(
        user_input_ids       = user_ids,
        user_attention_mask  = user_mask,
        route_descriptors    = obs.route_descriptors.unsqueeze(0).to(device),
        vehicle_descriptors  = obs.vehicle_descriptors.unsqueeze(0).to(device),
        pedestrian_descriptors = obs.pedestrian_descriptors.unsqueeze(0).to(device),
        ego_vehicle_descriptor = obs.ego_vehicle_descriptor.unsqueeze(0).to(device),
        max_length           = 64,

    )
    output_ids = gen_ids[0][enc.input_ids.shape[1]:]  # slice out only the new tokens
    answer: str = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(answer)

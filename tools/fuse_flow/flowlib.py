"""fuse_flow standalone harness: load BAGEL-7B-MoT on one GPU exactly as the rollout sees it (bf16 params, fp32
buffers, fp32 VAE) and build realistic gen / edit contexts (B=4, 512x512, think-length prefixes).

Used only for kernel-level profiling and operator accuracy tests; never a timing gate.
"""
import os
import sys
import time
from copy import deepcopy

import torch

MODEL = os.environ.get('MODEL_PATH', './checkpoints/BAGEL-7B-MoT')

PROMPTS = [
    'a photo of a red apple to the left of a blue cup',
    'a photo of three dogs and a green bicycle',
    'a photo of a purple umbrella above a white chair',
    'a photo of a yellow clock below an orange vase',
]
THINK = ('<think>The image should show the objects clearly on a simple background with natural lighting. '
         'I will place each object according to the described spatial relation, keep the colors saturated and '
         'distinct, use a centered composition with enough space between the objects so that each one is easy '
         'to recognise, and make the textures realistic. The background is a plain light-grey table surface '
         'with soft shadows, photographed at eye level with a 50mm lens.</think>')
EDIT_OPS = [
    'change the color of the cup to green',
    'add a second bicycle next to the dogs',
    'move the umbrella to the right side of the chair',
    'make the vase smaller and blue',
]


def setup(tree):
    tree = os.path.abspath(tree)
    os.chdir(tree)
    if tree not in sys.path:
        sys.path.insert(0, tree)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def load(tree, device='cuda'):
    setup(tree)
    from safetensors import safe_open
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.autoencoder import load_ae
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.diffusion.sde_sampler import SDESampler
    from train.rollout_controller import MultiRoundRolloutController

    t0 = time.time()
    llm_config = Qwen2Config.from_pretrained(MODEL + '/llm_config.json')
    llm_config.layer_module = 'Qwen2MoTDecoderLayer'
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    vit_config = SiglipVisionConfig.from_pretrained(MODEL + '/vit_config.json')
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + (-2)
    vit_config.rope = False
    torch.set_default_dtype(torch.bfloat16)   # params are bf16 during rollout (FSDP2 param_dtype); buffers explicit fp32
    with torch.device(device):
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
    torch.set_default_dtype(torch.float32)
    vae_model, vae_config = load_ae(local_path=MODEL + '/ae.safetensors')
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                         vae_config=vae_config, latent_patch_size=2, max_latent_size=64,
                         vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh', interpolate_pos=False,
                         timestep_shift=3.0)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = Bagel(language_model, vit_model, config)
    torch.set_default_dtype(torch.float32)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL)
    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)
    if num_new > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    params = dict(model.named_parameters())
    loaded = 0
    with safe_open(MODEL + '/ema.safetensors', 'pt', device=str(torch.device(device))) as f:
        for k in f.keys():
            if k in ('latent_pos_embed.pos_embed', 'vit_pos_embed.pos_embed'):
                continue
            if k in params:
                with torch.no_grad():
                    params[k].copy_(f.get_tensor(k))
                loaded += 1
    for n, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, n
        p.requires_grad_(False)
    for n, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != torch.float32:
            print('WARN buffer dtype', n, b.dtype)
    vae_model.to(device).eval()
    model.eval()
    vae_transform = ImageTransform(max_image_size=512, min_image_size=256, image_stride=16)
    vit_transform = ImageTransform(max_image_size=504, min_image_size=252, image_stride=14)
    sde_gen = SDESampler(eta_mode='monotonic', constant_eta=0.7, model_output_type='velocity')
    sde_edit = SDESampler(eta_mode='monotonic', constant_eta=1.0, model_output_type='velocity')
    ctrl = MultiRoundRolloutController(model=model, vae_model=vae_model, tokenizer=tokenizer,
                                       vae_transform=vae_transform, vit_transform=vit_transform,
                                       new_token_ids=new_token_ids, sde_sampler=sde_gen)
    print(f'[flowlib] loaded {loaded} tensors in {time.time() - t0:.1f}s', flush=True)
    return dict(model=model, vae=vae_model, ctrl=ctrl, tok=tokenizer, ids=new_token_ids, sde_gen=sde_gen,
                sde_edit=sde_edit)


def gen_contexts(env, B=4):
    from train.rollout_controller import GEN_THINK_SYSTEM_PROMPT
    ctrl = env['ctrl']
    prompts = (PROMPTS * B)[:B]
    max_cache_len = ctrl.estimate_gen_image_cache_len(prompts, (512, 512), True, 196)
    gen_context = ctrl.init_gen_context(B, multi_seq=True, use_static_cache=True, max_cache_len=max_cache_len)
    cfg_img_context = deepcopy(gen_context)
    gen_context = ctrl.update_context_text([GEN_THINK_SYSTEM_PROMPT] * B, gen_context)
    cfg_text_context = deepcopy(gen_context)
    gen_context = ctrl.update_context_text(prompts, gen_context)
    gen_context = ctrl.update_context_text([THINK] * B, gen_context)
    return gen_context, cfg_text_context, cfg_img_context


def run_gen(env, seed=1, B=4, sde_idx=5):
    ctrl = env['ctrl']
    ctrl.sde_sampler = env['sde_gen']
    g = torch.Generator(device='cuda'); g.manual_seed(seed)
    noise = torch.randn(1024, 64, dtype=torch.bfloat16, generator=g, device='cuda')
    gc, ct, ci = gen_contexts(env, B)
    with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        img, latents, log_probs, timesteps, dts = ctrl.gen_image(
            [(512, 512)] * B, gc, batch_size=B, cfg_text_precontext=ct, cfg_img_precontext=ci, cfg_text_scale=4.0,
            cfg_img_scale=1, cfg_interval=(0.4, 1.0), timestep_shift=3.0, num_timesteps=16, cfg_renorm_min=0.0,
            cfg_renorm_type='global', initial_noise=noise, enable_sde=True, sde_timestep_idx=[sde_idx])
    return img, latents, log_probs


def run_edit(env, imgs, seed=2, B=4, sde_idx=1):
    from PIL import Image
    ctrl = env['ctrl']
    ctrl.sde_sampler = env['sde_edit']
    g = torch.Generator(device='cuda'); g.manual_seed(seed)
    noise = torch.randn(1024, 64, dtype=torch.bfloat16, generator=g, device='cuda')
    input_imgs = [Image.fromarray(imgs[0])] * B
    ops = (EDIT_OPS * B)[:B]
    with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        img, out, extra = ctrl.edit_image(
            round_idx=1, step_idx=1, input_imgs=input_imgs, edit_operations=ops, think=True,
            cfg_text_scale=4.0, cfg_img_scale=1.5, cfg_interval=(0.4, 1.0), timestep_shift=3.0,
            num_timesteps=21, cfg_renorm_min=0.0, cfg_renorm_type='text_channel', initial_noise=noise,
            enable_sde=True, sde_timestep_idx=[sde_idx])
    return img, out

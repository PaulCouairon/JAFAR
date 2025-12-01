import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline


class StableDiffusionWrapper(nn.Module):
    def __init__(
        self,
        name="Manojb/stable-diffusion-2-1-base",
        target_layer="up_blocks.3",
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.target_layer = target_layer
        self.name = name

        print(f"Loading Stable Diffusion Backbone: {name}...")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            name, 
            torch_dtype=torch.float16
        )
        
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(device).eval()
        
        # Freeze everything
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Pre-compute empty text embedding (unconditional conditioning)
        self.register_buffer("empty_text_embeds", self._get_empty_text_embeds())
        
        # Clean up text encoder to save VRAM
        del self.text_encoder
        del self.tokenizer
        del pipe
        torch.cuda.empty_cache()

        # Setup Hook
        self.features = None
        self._register_hook()
        
        # Configuration for JAFAR / Train loop
        # SD 2.1 up_blocks[3] has 320 channels
        self.embed_dim = 320 
        # VAE downsamples by factor of 8 (512px -> 64px latent)
        self.patch_size = 8 

    def _get_empty_text_embeds(self):
        text_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features = output
        
        # Navigate to the layer string, e.g., "up_blocks.3"
        parts = self.target_layer.split(".")
        module = self.unet
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        module.register_forward_hook(hook_fn)

    def preprocess(self, img):
        # img is [B, 3, H, W] in [0, 1]
        # SD expects [-1, 1]
        x = 2.0 * img - 1.0
        return x

    @torch.no_grad()
    def forward(self, img):
        """
        Args:
            img: Tensor [B, 3, H, W] in range [0, 1]
        Returns:
            features: Tensor [B, 320, H/8, W/8]
            None: (No cls_token for SD)
        """
        # Ensure input is on correct device and dtype
        # We cast to float16 for the backbone forward pass
        x = self.preprocess(img).to(device=self.device, dtype=torch.float16)
        
        # 1. Encode to Latents
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        
        # 2. Forward UNet
        # Timestep 0 implies we are looking at the clean image structure
        t = torch.zeros(latents.shape[0], device=self.device, dtype=torch.float16)
        
        # We don't need the actual UNet output, just the hook execution
        _ = self.unet(
            latents, 
            t, 
            encoder_hidden_states=self.empty_text_embeds.repeat(latents.shape[0], 1, 1)
        )
        
        # Return features in float32 for stability in JAFAR loss calculation
        return self.features.float(), None
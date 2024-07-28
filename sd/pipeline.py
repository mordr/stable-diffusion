import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


# cfg = classifier free guidance
def generate(prompt: str, uncod_prompt: str, input_image=None, 
             strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name='ddmp', n_inference_steps=50, models={},
            seed=None,device=None,idle_device=None,tokenizer=None):
    
    with torch.no_grad():
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        if (not 0 < strength <= 1):
            raise ValueError('strength must be between 0 and 1')
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        clip = models['clip']
        clip.to(device)

        if do_cfg: # Classifier Free Guidance
            # Convert prompct to tokens using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids 
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens, None, to_idle)

            uncond_tokens = tokenizer.batch_encode_plus([uncod_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens, None, to_idle)

            # (2, seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids 
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens, None, to_idle)

        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Invalid Sampler Type!")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # Image-to-image
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor= input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)

            # (height, width, channel)
            input_image_tensor = torch.from_numpy(input_image_tensor).to(device).float()

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (batch_size, 4, height / 8, width / 8)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timestamp[0])

            to_idle(encoder)
        else:
            # Text-to-image, so start with random noise N(0, 1)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion'] 
        diffusion.to(device)

        timestamps = tqdm(sampler.timestamps)
        for i, timestamp in enumerate(timestamps):
            # (1, 320)
            time_embedding = get_time_embedding(timestamp).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents

            if do_cfg:
                # (batch_size, 4, latents_height, latents_width) -> (2 * batch_size, 4, latents_height, latents_weight)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by UNET
            latents = sampler.step(timestamp, latents, model_output)
        
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to(torch.uint8).cpu().numpy()
        return images[0]

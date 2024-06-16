import librosa
import torch
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
import os

from inferencer.base_inferencer import BaseInferencer
from inferencer.inferencer_full_band import full_band_no_truncation


@torch.no_grad()
def inference_wrapper(
        dataloader,
        model,
        device,
        inference_args,
        enhanced_dir=''
):
    for noisy, clean, _, name in tqdm(dataloader, desc="Inference"): # 언패킹 값 맞춰준거 맞춰주기
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]

        noisy = noisy.numpy().reshape(-1)

        if inference_args["inference_type"] == "full_band_no_truncation":
            noisy, enhanced = full_band_no_truncation(model, device, inference_args, noisy)
        else:
            raise NotImplementedError(f"Not implemented Inferencer type: {inference_args['inference_type']}")
        
        print(os.getcwd())
        ENHANCED_DIR_PA = Path("enhanced/enhanced") 
        ENHANCED_FILE_PA = Path(str(f"{name}.wav"))
        ENHANCED_PATH = ENHANCED_DIR_PA / ENHANCED_FILE_PA
        # sf.write("enhanced_dir" / f"{name}.wav", enhanced, 16000)
        
        print(enhanced_dir, name, ENHANCED_DIR_PA, ENHANCED_FILE_PA, ENHANCED_PATH)
        sf.write(ENHANCED_PATH, enhanced, 16000)

class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir
        )

import os
import tempfile
import shutil
import torch.utils
import numpy as np
from PIL import Image
from torch.autograd import Variable
from cog import BasePredictor, Path, Input

from model import Finetunemodel
from multi_read_data import MemoryFriendlyLoader


class Predictor(BasePredictor):
    def setup(self):
        self.models = {
            "easy": Finetunemodel("weights/easy.pt"),
            "medium": Finetunemodel("weights/medium.pt"),
            "difficult": Finetunemodel("weights/difficult.pt"),
        }
        for model in self.models.values():
            model = model.cuda()
            model.eval()

    def predict(
        self,
        image: Path = Input(
            description="Input low light image.",
        ),
        model_type: str = Input(
            choices=["easy", "medium", "difficult"],
            default="medium",
            description="Choose a model.",
        ),
    ) -> Path:
        input_dir = "cog_temp"
        os.makedirs(input_dir, exist_ok=True)
        try:
            model = self.models[model_type]

            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)

            TestDataset = MemoryFriendlyLoader(img_dir=input_dir, task="test")

            test_queue = torch.utils.data.DataLoader(
                TestDataset, batch_size=1, pin_memory=True, num_workers=0
            )

            output_path = Path(tempfile.mkdtemp()) / "output.png"
            with torch.no_grad():
                for _, (input, image_name) in enumerate(test_queue):
                    input = Variable(input, volatile=True).cuda()
                    i, r = model(input)
                    image_numpy = r[0].cpu().float().numpy()
                    image_numpy = np.transpose(image_numpy, (1, 2, 0))
                    im = Image.fromarray(
                        np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
                    )
                    im.save(str(output_path))
        finally:
            shutil.rmtree(input_dir)

        return output_path

import json
import os
from ultralytics.yolo.v8.detect.loss import v8DetectionLoss
import torch

# Ruta al archivo de configuración RLHF (puedes modificar si la ruta cambia)
RLHF_CONFIG_PATH = "rlhf_config.json"

class RLHFLoss(v8DetectionLoss):
    """
    Pérdida personalizada para ajuste RLHF: aplica factor de reward/penalización
    proveniente de feedback humano. Ajusta la magnitud de la pérdida final.
    """
    def __init__(self, model):
        super().__init__(model)
        self.reward_factor = 1.0  # default
        self.load_reward_factor()

    def load_reward_factor(self):
        if os.path.exists(RLHF_CONFIG_PATH):
            try:
                with open(RLHF_CONFIG_PATH, "r") as f:
                    config = json.load(f)
                    self.reward_factor = float(config.get("reward_factor", 1.0))
            except Exception as e:
                print(f"⚠️ Could not read RLHF config. Using default reward_factor=1.0. Error: {e}")
                self.reward_factor = 1.0

    def __call__(self, preds, batch):
        # Carga reward_factor cada vez para que refleje updates recientes
        self.load_reward_factor()
        original_loss, loss_items = super().__call__(preds, batch)
        # Modifica la pérdida con el factor extraído del feedback RLHF
        modified_loss = original_loss * self.reward_factor

        if isinstance(loss_items, torch.Tensor):
            loss_items_dict = {
                "box_loss": loss_items[0],
                "cls_loss": loss_items[1],
                "dfl_loss": loss_items[2],
            }
        else:
            loss_items_dict = loss_items

        loss_items_dict["reward_factor"] = self.reward_factor
        loss_items_dict["original_loss"] = original_loss.item()
        loss_items_dict["modified_loss"] = modified_loss.item()

        return modified_loss, loss_items_dict

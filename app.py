import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import pytorch_lightning as pl
from PIL import Image


# 1. Define Model Architecture 
class AgeGenderModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, loss_weight_age=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = models.efficientnet_v2_s(weights=None) # loading weights from .ckpt
        
        num_features = self.backbone.classifier[1].in_features 
        
        self.backbone.classifier[1] = nn.Identity() 

        self.age_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1) 
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        features = self.backbone(x)
        age_pred = self.age_head(features).squeeze(-1)
        gender_pred = self.gender_head(features).squeeze(-1)
        return {"age": age_pred, "gender": gender_pred}


# 2. Setup Transform and Load Model
IMG_SIZE = 300

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading model...")
model = AgeGenderModel.load_from_checkpoint("model.ckpt", map_location=torch.device('cpu'))
model.eval()


# 3. Define Inference Function
def predict(image):
    if image is None:
        return "Please upload an image.", None
    
    # 1. Transform Image
    img_tensor = transform(image).unsqueeze(0) # Add batch dimension -> (1, 3, 300, 300)
    
    # 2. Inference
    with torch.no_grad():
        preds = model(img_tensor)
        
    # 3. Post-processing
    # Age
    age_prediction = preds["age"].item()
    age_prediction = max(0, round(age_prediction, 1)) # Ensure non-negative and round it
    
    # Gender
    gender_logit = preds["gender"].item()
    gender_prob = torch.sigmoid(torch.tensor(gender_logit)).item()
    
    if gender_prob > 0.5:
        gender_label = "Male" 
    else:
        gender_label = "Female"
        
    return age_prediction, gender_label


# 4. Create Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Number(label="Predicted Age"),
        gr.Textbox(label="Predicted Gender")
    ],
    title="Age & Gender Prediction App",
    description="Upload a photo of a face to predict age and gender."
)

if __name__ == "__main__":
    interface.launch()
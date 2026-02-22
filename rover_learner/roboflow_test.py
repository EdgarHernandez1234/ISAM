
from inference import get_model
import supervision as sv

# Get your model ID and API key from the "Deploy" page in the Roboflow dashboard
model = get_model(model_id="object-detection-in-sand/1", api_key="JagcRj32ljVtLqGHoRKfy")

# Infer on an image (can be a local path, numpy array, or URL)
results = model.infer("WhatsApp Image 2025-06-18 at 18.11.54.jpeg")[0]
results = model.infer("images (27).jpg")[0]

# Process results with Supervision (optional)
detections = sv.Detections.from_inference(results)
# ... further processing or visualization

*if that api key doesnt work, use: rf_0fJm5jm2AXYfPISiCOuiZeRCj2p2

